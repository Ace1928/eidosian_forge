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
def print_table(self, stream=sys.stdout, filter_function=None):
    """
        A pretty ASCII printer for the periodic table, based on some filter_function.

        Args:
            stream: file-like object
            filter_function:
                A filtering function that take a Pseudo as input and returns a boolean.
                For example, setting filter_function = lambda p: p.Z_val > 2 will print
                a periodic table containing only pseudos with Z_val > 2.
        """
    print(self.to_table(filter_function=filter_function), file=stream)