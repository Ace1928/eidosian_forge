from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
def identify_potcar(self, mode: Literal['data', 'file']='data', data_tol: float=1e-06) -> tuple[list[str], list[str]]:
    """
        Identify the symbol and compatible functionals associated with this PotcarSingle.

        This method checks the summary statistics of either the POTCAR metadadata
        (PotcarSingle._summary_stats[key]["header"] for key in ("keywords", "stats") )
        or the entire POTCAR file (PotcarSingle._summary_stats) against a database
        of hashes for POTCARs distributed with VASP 5.4.4.

        Args:
            mode ('data' | 'file'): 'data' mode checks the POTCAR header keywords and stats only
                while 'file' mode checks the entire summary stats.
            data_tol (float): Tolerance for comparing the summary statistics of the POTCAR
                with the reference statistics.

        Returns:
            symbol (list): List of symbols associated with the PotcarSingle
            potcar_functionals (list): List of potcar functionals associated with
                the PotcarSingle
        """
    if mode == 'data':
        check_modes = ['header']
    elif mode == 'file':
        check_modes = ['header', 'data']
    else:
        raise ValueError(f"Bad mode={mode!r}. Choose 'data' or 'file'.")
    identity: dict[str, list] = {'potcar_functionals': [], 'potcar_symbols': []}
    for func in self.functional_dir:
        for ref_psp in self._potcar_summary_stats[func].get(self.TITEL.replace(' ', ''), []):
            if self.VRHFIN.replace(' ', '') != ref_psp['VRHFIN']:
                continue
            key_match = all((set(ref_psp['keywords'][key]) == set(self._summary_stats['keywords'][key]) for key in check_modes))
            data_diff = [abs(ref_psp['stats'][key][stat] - self._summary_stats['stats'][key][stat]) for stat in ['MEAN', 'ABSMEAN', 'VAR', 'MIN', 'MAX'] for key in check_modes]
            data_match = all(np.array(data_diff) < data_tol)
            if key_match and data_match:
                identity['potcar_functionals'].append(func)
                identity['potcar_symbols'].append(ref_psp['symbol'])
    for key, values in identity.items():
        if len(values) == 0:
            return ([], [])
        identity[key] = list(set(values))
    return (identity['potcar_functionals'], identity['potcar_symbols'])