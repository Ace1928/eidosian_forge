import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
def electronic_me_Qcc(self, omega, gamma):
    self.read()
    Vel_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
    approximation = self.approximation.lower()
    if approximation == 'profeta':
        Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma)
    elif approximation == 'placzek':
        Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, True)
    elif approximation == 'p-p':
        Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, -1)
    else:
        raise RuntimeError('Bug: call with {0} should not happen!'.format(self.approximation))
    return self.map_to_modes(Vel_rcc)