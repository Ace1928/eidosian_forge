from __future__ import annotations
import abc
import json
import logging
import os
import warnings
from glob import glob
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.gaussian import GaussianOutput
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar
from pymatgen.io.vasp.outputs import Dynmat, Oszicar, Vasprun
def assimilate(self, path):
    """Assimilate data in a directory path into a ComputedEntry object.

        Args:
            path: directory path

        Returns:
            ComputedEntry
        """
    try:
        gau_run = GaussianOutput(path)
    except Exception as exc:
        logger.debug(f'error in {path}: {exc}')
        return None
    param = {}
    for p in self._parameters:
        param[p] = getattr(gau_run, p)
    data = {}
    for d in self._data:
        data[d] = getattr(gau_run, d)
    if self._inc_structure:
        entry = ComputedStructureEntry(gau_run.final_structure, gau_run.final_energy, parameters=param, data=data)
    else:
        entry = ComputedEntry(gau_run.final_structure.composition, gau_run.final_energy, parameters=param, data=data)
    return entry