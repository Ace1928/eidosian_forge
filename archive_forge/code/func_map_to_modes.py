import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
def map_to_modes(self, V_rcc):
    V_qcc = (V_rcc.T * self.im_r).T
    V_Qcc = np.dot(V_qcc.T, self.modes_Qq.T).T
    return V_Qcc