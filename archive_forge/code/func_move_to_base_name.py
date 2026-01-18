import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def move_to_base_name(self, basename):
    """ when output tracking is on or the base namem is not standard,
        this routine will rename add the base to the cube file output for
        easier tracking """
    for plot in self.plots:
        found = False
        cube = plot.split()
        if cube[0] == 'total_density' or cube[0] == 'spin_density' or cube[0] == 'delta_density':
            found = True
            old_name = cube[0] + '.cube'
            new_name = basename + '.' + old_name
        if cube[0] == 'eigenstate' or cube[0] == 'eigenstate_density':
            found = True
            state = int(cube[1])
            s_state = cube[1]
            for i in [10, 100, 1000, 10000]:
                if state < i:
                    s_state = '0' + s_state
            old_name = cube[0] + '_' + s_state + '_spin_1.cube'
            new_name = basename + '.' + old_name
        if found:
            os.system('mv ' + old_name + ' ' + new_name)