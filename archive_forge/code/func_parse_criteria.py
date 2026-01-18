from __future__ import annotations
import itertools
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
from enum import Enum, unique
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import requests
from monty.json import MontyDecoder, MontyEncoder
from ruamel.yaml import YAML
from tqdm import tqdm
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.exp_entries import ExpEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
@staticmethod
def parse_criteria(criteria_string):
    """Parses a powerful and simple string criteria and generates a proper
        mongo syntax criteria.

        Args:
            criteria_string (str): A string representing a search criteria.
                Also supports wild cards. E.g.,
                something like "*2O" gets converted to
                {'pretty_formula': {'$in': [u'B2O', u'Xe2O', u"Li2O", ...]}}

                Other syntax examples:
                    mp-1234: Interpreted as a Materials ID.
                    Fe2O3 or *2O3: Interpreted as reduced formulas.
                    Li-Fe-O or *-Fe-O: Interpreted as chemical systems.

                You can mix and match with spaces, which are interpreted as
                "OR". E.g., "mp-1234 FeO" means query for all compounds with
                reduced formula FeO or with materials_id mp-1234.

        Returns:
            A mongo query dict.
        """
    tokens = criteria_string.split()

    def parse_sym(sym):
        if sym == '*':
            return [el.symbol for el in Element]
        m = re.match('\\{(.*)\\}', sym)
        if m:
            return [s.strip() for s in m.group(1).split(',')]
        return [sym]

    def parse_tok(t):
        if re.match('\\w+-\\d+', t):
            return {'task_id': t}
        if '-' in t:
            elements = [parse_sym(sym) for sym in t.split('-')]
            chem_sys_lst = []
            for cs in itertools.product(*elements):
                if len(set(cs)) == len(cs):
                    cs = [Element(s).symbol for s in cs]
                    chem_sys_lst.append('-'.join(sorted(cs)))
            return {'chemsys': {'$in': chem_sys_lst}}
        all_formulas = set()
        explicit_els = []
        wild_card_els = []
        for sym in re.findall('(\\*[\\.\\d]*|\\{.*\\}[\\.\\d]*|[A-Z][a-z]*)[\\.\\d]*', t):
            if '*' in sym or '{' in sym:
                wild_card_els.append(sym)
            else:
                m = re.match('([A-Z][a-z]*)[\\.\\d]*', sym)
                explicit_els.append(m.group(1))
        n_elements = len(wild_card_els) + len(set(explicit_els))
        parts = re.split('(\\*|\\{.*\\})', t)
        parts = [parse_sym(s) for s in parts if s != '']
        for formula in itertools.product(*parts):
            comp = Composition(''.join(formula))
            if len(comp) == n_elements:
                for elem in comp:
                    Element(elem.symbol)
                all_formulas.add(comp.reduced_formula)
        return {'pretty_formula': {'$in': list(all_formulas)}}
    if len(tokens) == 1:
        return parse_tok(tokens[0])
    return {'$or': list(map(parse_tok, tokens))}