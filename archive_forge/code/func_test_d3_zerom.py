import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_d3_zerom(factory, system):
    system.calc = factory.calc(damping='zerom')
    close(system.get_potential_energy(), -2.4574447613705717)