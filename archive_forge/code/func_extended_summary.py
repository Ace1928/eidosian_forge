import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def extended_summary(self, omega=0, gamma=0, method='standard', direction='central', log=sys.stdout):
    """Print summary for given omega [eV]"""
    self.read(method, direction)
    om_v = self.get_energies()
    intens_v = self.intensity(omega, gamma)
    if isinstance(log, str):
        log = paropen(log, 'a')
    parprint('-------------------------------------', file=log)
    parprint(' excitation at ' + str(omega) + ' eV', file=log)
    parprint(' gamma ' + str(gamma) + ' eV', file=log)
    parprint(' approximation:', self.approximation, file=log)
    parprint(' observation:', self.observation, file=log)
    parprint(' Mode    Frequency        Intensity', file=log)
    parprint('  #    meV     cm^-1      [e^4A^4/eV^2]', file=log)
    parprint('-------------------------------------', file=log)
    for v, e in enumerate(om_v):
        parprint(self.ind_v[v], '{0:6.1f}   {1:7.1f} {2:9.1f}'.format(1000 * e, e / u.invcm, 1000000000.0 * intens_v[v]), file=log)
    parprint('-------------------------------------', file=log)
    parprint('Zero-point energy: %.3f eV' % self.vibrations.get_zero_point_energy(), file=log)