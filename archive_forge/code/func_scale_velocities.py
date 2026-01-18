import numpy as np
from ase.md.md import MolecularDynamics
from ase.parallel import world
def scale_velocities(self):
    """ Do the NVT Berendsen velocity scaling """
    tautscl = self.dt / self.taut
    old_temperature = self.atoms.get_temperature()
    scl_temperature = np.sqrt(1.0 + (self.temperature / old_temperature - 1.0) * tautscl)
    if scl_temperature > 1.1:
        scl_temperature = 1.1
    if scl_temperature < 0.9:
        scl_temperature = 0.9
    p = self.atoms.get_momenta()
    p = scl_temperature * p
    self.atoms.set_momenta(p)
    return