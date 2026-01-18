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
def get_pseudos_for_structure(self, structure: Structure):
    """
        Return the list of Pseudo objects to be used for this Structure.

        Args:
            structure: pymatgen Structure.

        Raises:
            `ValueError` if one of the chemical symbols is not found or
            multiple occurrences are present in the table.
        """
    return self.pseudos_with_symbols(structure.symbol_set)