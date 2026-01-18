from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
def select_symbols(self, symbols, ret_list=False):
    """
        Return a PseudoTable with the pseudopotentials with the given list of chemical symbols.

        Args:
            symbols: str or list of symbols
                Prepend the symbol string with "-", to exclude pseudos.
            ret_list: if True a list of pseudos is returned instead of a PseudoTable
        """
    if isinstance(symbols, str):
        symbols = [symbols]
    exclude = symbols[0].startswith('-')
    if exclude:
        if not all((s.startswith('-') for s in symbols)):
            raise ValueError('When excluding symbols, all strings must start with `-`')
        symbols = [s[1:] for s in symbols]
    symbols = set(symbols)
    pseudos = []
    for p in self:
        if exclude:
            if p.symbol in symbols:
                continue
        elif p.symbol not in symbols:
            continue
        pseudos.append(p)
    if ret_list:
        return pseudos
    return type(self)(pseudos)