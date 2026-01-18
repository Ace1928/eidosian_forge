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
def oncvpsp_header(filename, ppdesc):
    """
        Parse the ONCVPSP abinit header. Example:

        Li    ONCVPSP  r_core=  2.01  3.02
              3.0000      3.0000      140504    zatom,zion,pspd
             8     2     1     4   600     0    pspcod,pspxc,lmax,lloc,mmax,r2well
          5.99000000  0.00000000  0.00000000    rchrg fchrg qchrg
             2     2     0     0     0    nproj
             0                 extension_switch
           0                        -2.5000025868368D+00 -1.2006906995331D+00
             1  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
             2  1.0000000000000D-02  4.4140499497377D-02  1.9909081701712D-02
        """
    lines = _read_nlines(filename, 6)
    header = _dict_from_lines(lines[:3], [0, 3, 6])
    summary = lines[0]
    header['pspdat'] = header['pspd']
    header.pop('pspd')
    header['extension_switch'] = int(lines[5].split()[0])
    return NcAbinitHeader(summary, **header)