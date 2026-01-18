from numpy import random, cos, pi, log, ones, repeat
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
def boltzmann_random(self, width, size):
    x = self.rng.random_sample(size=size)
    y = self.rng.random_sample(size=size)
    z = width * cos(2 * pi * x) * (-2 * log(1 - y)) ** 0.5
    return z