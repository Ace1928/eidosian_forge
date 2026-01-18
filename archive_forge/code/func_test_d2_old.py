import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_d2_old(factory, system):
    system.calc = factory.calc(old=True)
    close(system.get_potential_energy(), -0.8923443424663762)