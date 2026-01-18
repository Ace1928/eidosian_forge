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
def submit_structures(self, structures, authors, projects=None, references='', remarks=None, data=None, histories=None, created_at=None):
    """Submits a list of structures to the Materials Project as SNL files.
        The argument list mirrors the arguments for the StructureNL object,
        except that a list of structures with the same metadata is used as an
        input.

        Note:
            As of now, this MP REST feature is open only to a select group of
            users. Opening up submissions to all users is being planned for the future.

        Args:
            structures: A list of Structure objects
            authors (list): List of {"name":'', "email":''} dicts,
                *list* of Strings as 'John Doe <johndoe@gmail.com>',
                or a single String with commas separating authors
            projects ([str]): List of Strings ['Project A', 'Project B'].
                This applies to all structures.
            references (str): A String in BibTeX format. Again, this applies to
                all structures.
            remarks ([str]): List of Strings ['Remark A', 'Remark B']
            data ([dict]): A list of free form dict. Namespaced at the root
                level with an underscore, e.g. {"_materialsproject":<custom
                data>}. The length of data should be the same as the list of
                structures if not None.
            histories: List of list of dicts - [[{'name':'', 'url':'',
                'description':{}}], ...] The length of histories should be the
                same as the list of structures if not None.
            created_at (datetime): A datetime object

        Returns:
            A list of inserted submission ids.
        """
    from pymatgen.util.provenance import StructureNL
    snl_list = StructureNL.from_structures(structures, authors, projects, references, remarks, data, histories, created_at)
    self.submit_snl(snl_list)