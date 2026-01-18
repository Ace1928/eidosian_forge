import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
def systems_bulk():
    atoms = bulk('Ar', cubic=True)
    atoms.set_cell(atoms.cell * stretch, scale_atoms=True)
    calc = LennardJones(rc=10)
    atoms.calc = calc
    yield atoms
    atoms = bulk('Ar', cubic=True)
    atoms.set_cell(atoms.cell * stretch, scale_atoms=True)
    calc = LennardJones(rc=12, ro=10, smooth=True)
    atoms.calc = calc
    yield atoms