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
def infos_on_basis_set(self):
    return f'=========================================\nReading basis set:\n\nBasis set for {self.filename} atom \nMaximum angular momentum = {self.data['lmax']}\nNumber of atomics orbitals = {self.data['n_nlo']}\nNumber of nlm orbitals = {self.data['n_nlmo']}\n========================================='