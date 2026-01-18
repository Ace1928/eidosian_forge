from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def set_species_cond(self, spc):
    """ Sets the conduction species in the current dat instance,
        in onetep this includes both atomic number information
        as well as NGWF parameters like number and cut off radius"""
    self.species_cond = deepcopy(spc)