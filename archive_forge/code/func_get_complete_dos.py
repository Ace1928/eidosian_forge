from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def get_complete_dos(self, structure: Structure, analyzer_for_second_spin=None):
    """Gives a CompleteDos object with the DOS from the interpolated projected band structure.

        Args:
            structure: necessary to identify sites for projection
            analyzer_for_second_spin: must be specified to have a CompleteDos with both Spin components

        Returns:
            CompleteDos: from the interpolated projected band structure

        Example of use in case of spin polarized case:

            BoltztrapRunner(bs=bs,nelec=10,run_type="DOS",spin=1).run(path_dir='dos_up/')
            an_up=BoltztrapAnalyzer.from_files("dos_up/boltztrap/",dos_spin=1)

            BoltztrapRunner(bs=bs,nelec=10,run_type="DOS",spin=-1).run(path_dir='dos_dw/')
            an_dw=BoltztrapAnalyzer.from_files("dos_dw/boltztrap/",dos_spin=-1)

            cdos=an_up.get_complete_dos(bs.structure,an_dw)
        """
    pdoss: dict[PeriodicSite, dict[Orbital, dict[Spin, ArrayLike]]] = {}
    spin_1 = next(iter(self.dos.densities))
    if analyzer_for_second_spin:
        if not np.all(self.dos.energies == analyzer_for_second_spin.dos.energies):
            raise BoltztrapError('Dos merging error: energies of the two dos are different')
        spin_2 = next(iter(analyzer_for_second_spin.dos.densities))
        if spin_1 == spin_2:
            raise BoltztrapError('Dos merging error: spin component are the same')
    for s in self._dos_partial:
        idx = int(s)
        if structure[idx] not in pdoss:
            pdoss[structure[idx]] = {}
        for o in self._dos_partial[s]:
            if Orbital[o] not in pdoss[structure[idx]]:
                pdoss[structure[idx]][Orbital[o]] = {}
            pdoss[structure[idx]][Orbital[o]][spin_1] = self._dos_partial[s][o]
            if analyzer_for_second_spin:
                pdoss[structure[idx]][Orbital[o]][spin_2] = analyzer_for_second_spin._dos_partial[s][o]
    if analyzer_for_second_spin:
        total_dos = Dos(self.dos.efermi, self.dos.energies, {spin_1: self.dos.densities[spin_1], spin_2: analyzer_for_second_spin.dos.densities[spin_2]})
    else:
        total_dos = self.dos
    return CompleteDos(structure, total_dos=total_dos, pdoss=pdoss)