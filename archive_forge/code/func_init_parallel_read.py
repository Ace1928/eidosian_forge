import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
def init_parallel_read(self):
    """Initialize variables for parallel read"""
    rank = self.comm.rank
    indices = self.indices
    myn = -(-self.ndof // self.comm.size)
    self.slize = s = slice(myn * rank, myn * (rank + 1))
    self.myindices = np.repeat(indices, 3)[s]
    self.myxyz = ('xyz' * len(indices))[s]
    self.myr = range(self.ndof)[s]
    self.mynd = len(self.myr)