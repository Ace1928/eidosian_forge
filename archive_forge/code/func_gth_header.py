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
def gth_header(filename, ppdesc):
    """
        Parse the GTH abinit header. Example:

        Goedecker-Teter-Hutter  Wed May  8 14:27:44 EDT 1996
        1   1   960508                     zatom,zion,pspdat
        2   1   0    0    2001    0.       pspcod,pspxc,lmax,lloc,mmax,r2well
        0.2000000 -4.0663326  0.6778322 0 0     rloc, c1, c2, c3, c4
        0 0 0                              rs, h1s, h2s
        0 0                                rp, h1p
          1.36 .2   0.6                    rcutoff, rloc
        """
    lines = _read_nlines(filename, 7)
    header = _dict_from_lines(lines[:3], [0, 3, 6])
    summary = lines[0]
    return NcAbinitHeader(summary, **header)