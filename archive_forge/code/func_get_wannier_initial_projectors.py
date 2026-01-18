import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_wannier_initial_projectors(atoms, parameters):
    """
    B. Specify the orbital, central position and orientation of a projector
    Wannier.Initial.Projectos will be used to specify the projector name,
    local orbital function, center of local orbital, and the local z-axis and
    x-axis for orbital orientation.

    An example setting is shown here:
    wannier_initial_projectors=
    [['proj1-sp3','0.250','0.250','0.25','-1.0','0.0','0.0','0.0','0.0','-1.0']
    ,['proj1-sp3','0.000','0.000','0.00','0.0','0.0','1.0','1.0','0.0','0.0']]
    Goes to,
        <Wannier.Initial.Projectors
           proj1-sp3   0.250  0.250  0.250   -1.0 0.0 0.0    0.0  0.0 -1.0
           proj1-sp3   0.000  0.000  0.000    0.0 0.0 1.0    1.0  0.0  0.0
        Wannier.Initial.Projectors>
    """
    return parameters.get('wannier_initial_projectors', [])