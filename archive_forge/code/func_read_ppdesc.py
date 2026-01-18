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
def read_ppdesc(self, filename):
    """
        Read the pseudopotential descriptor from filename.

        Returns:
            Pseudopotential descriptor. None if filename is not a valid pseudopotential file.

        Raises:
            `PseudoParseError` if fileformat is not supported.
        """
    if filename.endswith('.xml'):
        raise self.Error('XML pseudo not supported yet')
    lines = _read_nlines(filename, 80)
    for lineno, line in enumerate(lines, start=1):
        if lineno == 3:
            try:
                tokens = line.split()
                pspcod, _pspxc = map(int, tokens[:2])
            except Exception:
                msg = f'{filename}: Cannot parse pspcod, pspxc in line\n {line}'
                logger.critical(msg)
                return None
            if pspcod not in self._PSPCODES:
                raise self.Error(f"{filename}: Don't know how to handle pspcod={pspcod!r}\n")
            ppdesc = self._PSPCODES[pspcod]
            if pspcod == 7:
                tokens = lines[lineno].split()
                pspfmt, _creatorID = tokens[:2]
                ppdesc = ppdesc._replace(format=pspfmt)
            return ppdesc
    return None