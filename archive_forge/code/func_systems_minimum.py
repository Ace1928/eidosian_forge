import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
def systems_minimum():
    """two atoms at potential minimum"""
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 2 ** (1.0 / 6.0)]])
    calc = LennardJones(rc=100000.0)
    atoms.calc = calc
    yield atoms
    calc = LennardJones(rc=100000.0, smooth=True)
    atoms.calc = calc
    yield atoms