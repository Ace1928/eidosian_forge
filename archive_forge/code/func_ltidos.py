from math import pi, sqrt
import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.parallel import world
from ase.utils.cext import cextension
def ltidos(*args, **kwargs):
    raise DeprecationWarning('Please use linear_tetrahedron_integration().')