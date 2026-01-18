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
def pseudo_with_symbol(self, symbol, allow_multi=False):
    """
        Return the pseudo with the given chemical symbol.

        Args:
            symbols: String with the chemical symbol of the element
            allow_multi: By default, the method raises ValueError
                if multiple occurrences are found. Use allow_multi to prevent this.

        Raises:
            ValueError if symbol is not found or multiple occurrences are present and not allow_multi
        """
    pseudos = self.select_symbols(symbol, ret_list=True)
    if not pseudos or (len(pseudos) > 1 and (not allow_multi)):
        raise ValueError(f'Found {len(pseudos)} occurrences of symbol={symbol!r}')
    if not allow_multi:
        return pseudos[0]
    return pseudos