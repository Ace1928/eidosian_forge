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
@lazy_property
def projector_functions(self):
    """Dictionary with the PAW projectors indexed by state."""
    projector_functions = {}
    for mesh, values, attrib in self._parse_all_radfuncs('projector_function'):
        state = attrib['state']
        projector_functions[state] = RadialFunction(mesh, values)
    return projector_functions