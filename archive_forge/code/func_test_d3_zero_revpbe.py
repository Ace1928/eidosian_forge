import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_d3_zero_revpbe(factory, system):
    system.calc = factory.calc(xc='revpbe')
    close(system.get_potential_energy(), -1.5274869363442936)