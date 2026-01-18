import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_d3_bjm(factory, system):
    system.calc = factory.calc(damping='bjm')
    close(system.get_potential_energy(), -1.4662085277005799)