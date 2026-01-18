import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def get_wannier_localization_matrix(self, nbands, dirG, nextkpoint=None, kpoint=None, spin=0, G_I=(0, 0, 0)):
    try:
        self['bloch_overlaps']
    except KeyError:
        self.read_bloch_overlaps()
    dirG = tuple(dirG)
    nx, ny, nz = self['wannier_kpts']
    nr3 = nx * ny * nz
    if kpoint is None and nextkpoint is None:
        return {kpoint: self['bloch_overlaps'][kpoint][dirG][:nbands, :nbands] for kpoint in range(nr3)}
    if kpoint is None:
        kpoint = (nextkpoint - self.dk(dirG)) % nr3
    if nextkpoint is None:
        nextkpoint = (kpoint + self.dk(dirG)) % nr3
    if dirG not in self['bloch_overlaps'][kpoint].keys():
        return np.zeros((nbands, nbands), complex)
    return self['bloch_overlaps'][kpoint][dirG][:nbands, :nbands]