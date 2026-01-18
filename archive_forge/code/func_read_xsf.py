import numpy as np
from ase.atoms import Atoms
from ase.units import Hartree
from ase.data import atomic_numbers
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import writer, reader
def read_xsf(fileobj, index=-1, read_data=False):
    images = list(iread_xsf(fileobj, read_data=read_data))
    if read_data:
        array = images[-1]
        images = images[:-1]
        return (array, images[index])
    return images[index]