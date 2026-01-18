import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_d3_bj_abc(factory, system):
    system.calc = factory.calc(damping='bj', abc=True)
    close(system.get_potential_energy(), -1.1959417763402416)