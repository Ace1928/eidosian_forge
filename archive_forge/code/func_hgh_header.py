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
@staticmethod
def hgh_header(filename, ppdesc):
    """
        Parse the HGH abinit header. Example:

        Hartwigsen-Goedecker-Hutter psp for Ne,  from PRB58, 3641 (1998)
            10   8  010605 zatom,zion,pspdat
            3 1   1 0 2001 0  pspcod,pspxc,lmax,lloc,mmax,r2well
        """
    lines = _read_nlines(filename, 3)
    header = _dict_from_lines(lines[:3], [0, 3, 6])
    summary = lines[0]
    return NcAbinitHeader(summary, **header)