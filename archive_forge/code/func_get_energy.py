import numpy as np
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory
def get_energy(self, positions):
    """Return the energy of the nearest local minimum."""
    if np.sometrue(self.positions != positions):
        self.positions = positions
        self.atoms.set_positions(positions)
        with self.optimizer(self.atoms, logfile=self.optimizer_logfile) as opt:
            opt.run(fmax=self.fmax)
        if self.lm_trajectory is not None:
            self.lm_trajectory.write(self.atoms)
        self.energy = self.atoms.get_potential_energy()
    return self.energy