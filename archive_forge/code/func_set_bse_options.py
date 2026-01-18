from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
def set_bse_options(self, n_excitations=10, nit_bse=200):
    """
        Set parameters in cell.in for a BSE computation

        Args:
            nv_bse: number of valence bands
            nc_bse: number of conduction bands
            n_excitations: number of excitations
            nit_bse: number of iterations.
        """
    self.bse_tddft_options.update(npsi_bse=n_excitations, nit_bse=nit_bse)