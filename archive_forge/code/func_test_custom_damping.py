import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_custom_damping(factory, system):
    system.calc = factory.calc(s6=1.1, sr6=1.1, s8=0.6, sr8=0.9, alpha6=13.0)
    close(system.get_potential_energy(), -1.082846357973487)