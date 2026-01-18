from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
@property
def cder(self):
    """Complex CDER from WAVEDER."""
    return self.cder_real + self.cder_imag * 1j