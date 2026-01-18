import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def set_point_charges(self, pcpot=None):
    """write external point charges to control"""
    if pcpot is not None and pcpot != self.pcpot:
        self.pcpot = pcpot
    if self.pcpot.mmcharges is None or self.pcpot.mmpositions is None:
        raise RuntimeError('external point charges not defined')
    if not self.pc_initialized:
        if len(read_data_group('point_charges')) == 0:
            add_data_group('point_charges', 'file=pc.txt')
        if len(read_data_group('point_charge_gradients')) == 0:
            add_data_group('point_charge_gradients', 'file=pc_gradients.txt')
        drvopt = read_data_group('drvopt')
        if 'point charges' not in drvopt:
            drvopt += '\n   point charges\n'
            delete_data_group('drvopt')
            add_data_group(drvopt, raw=True)
        self.pc_initialized = True
    if self.pcpot.updated:
        with open('pc.txt', 'w') as pcfile:
            pcfile.write('$point_charges nocheck list\n')
            for (x, y, z), charge in zip(self.pcpot.mmpositions, self.pcpot.mmcharges):
                pcfile.write('%20.14f  %20.14f  %20.14f  %20.14f\n' % (x / Bohr, y / Bohr, z / Bohr, charge))
            pcfile.write('$end \n')
        self.pcpot.updated = False