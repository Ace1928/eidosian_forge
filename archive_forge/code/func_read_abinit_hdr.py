from __future__ import annotations
import logging
import os.path
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.dev import requires
from monty.functools import lazy_property
from monty.string import marquee
from pymatgen.core.structure import Structure
from pymatgen.core.units import ArrayWithUnit
from pymatgen.core.xcfunc import XcFunc
def read_abinit_hdr(self):
    """
        Read the variables associated to the Abinit header.

        Return AbinitHeader
        """
    dct = {}
    for hvar in _HDR_VARIABLES.values():
        ncname = hvar.etsf_name if hvar.etsf_name is not None else hvar.name
        if ncname in self.rootgrp.variables:
            dct[hvar.name] = self.read_value(ncname)
        elif ncname in self.rootgrp.dimensions:
            dct[hvar.name] = self.read_dimvalue(ncname)
        else:
            raise ValueError(f'Cannot find `{ncname}` in `{self.path}`')
        if hasattr(dct[hvar.name], 'shape') and (not dct[hvar.name].shape):
            dct[hvar.name] = np.asarray(dct[hvar.name]).item()
        if hvar.name in ('title', 'md5_pseudos', 'codvsn'):
            if hvar.name == 'codvsn':
                dct[hvar.name] = ''.join((bs.decode('utf-8').strip() for bs in dct[hvar.name]))
            else:
                dct[hvar.name] = [''.join((bs.decode('utf-8') for bs in astr)).strip() for astr in dct[hvar.name]]
    return AbinitHeader(dct)