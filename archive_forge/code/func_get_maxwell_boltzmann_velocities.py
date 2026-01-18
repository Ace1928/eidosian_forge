from numpy import random, cos, pi, log, ones, repeat
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
def get_maxwell_boltzmann_velocities(self):
    natoms = len(self.atoms)
    masses = repeat(self.masses, 3).reshape(natoms, 3)
    width = (self.temp / masses) ** 0.5
    velos = self.boltzmann_random(width, size=(natoms, 3))
    return velos