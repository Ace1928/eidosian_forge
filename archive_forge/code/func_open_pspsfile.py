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
def open_pspsfile(self, ecut=20, pawecutdg=None):
    """
        Calls Abinit to compute the internal tables for the application of the
        pseudopotential part. Returns PspsFile object providing methods
        to plot and analyze the data or None if file is not found or it's not readable.

        Args:
            ecut: Cutoff energy in Hartree.
            pawecutdg: Cutoff energy for the PAW double grid.
        """
    from abipy.abio.factories import gs_input
    from abipy.core.structure import Structure
    from abipy.electrons.psps import PspsFile
    from abipy.flowtk import AbinitTask
    lattice = 10 * np.eye(3)
    structure = Structure(lattice, [self.element], coords=[[0, 0, 0]])
    if self.ispaw and pawecutdg is None:
        pawecutdg = ecut * 4
    inp = gs_input(structure, pseudos=[self], ecut=ecut, pawecutdg=pawecutdg, spin_mode='unpolarized', kppa=1)
    inp['prtpsps'] = -1
    task = AbinitTask.temp_shell_task(inp)
    task.start_and_wait()
    filepath = task.outdir.has_abiext('_PSPS.nc')
    if not filepath:
        logger.critical(f'Cannot find PSPS.nc file in {task.outdir}')
        return None
    try:
        return PspsFile(filepath)
    except Exception as exc:
        logger.critical(f'Exception while reading PSPS file at {filepath}:\n{exc}')
        return None