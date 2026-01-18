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
def all_combinations_for_elements(self, element_symbols):
    """
        Return a list with all the possible combination of pseudos
        for the given list of element_symbols.
        Each item is a list of pseudopotential objects.

        Example:
            table.all_combinations_for_elements(["Li", "F"])
        """
    dct = {}
    for symbol in element_symbols:
        dct[symbol] = self.select_symbols(symbol, ret_list=True)
    from itertools import product
    return list(product(*dct.values()))