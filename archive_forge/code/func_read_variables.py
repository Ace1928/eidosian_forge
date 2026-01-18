import gzip
import struct
from collections import deque
from os.path import splitext
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.quaternions import Quaternions
def read_variables(string):
    obj_len = struct.calcsize(string)
    data_obj = fileobj.read(obj_len)
    if obj_len != len(data_obj):
        raise EOFError
    return struct.unpack(string, data_obj)