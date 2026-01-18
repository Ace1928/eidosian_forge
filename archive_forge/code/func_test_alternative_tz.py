import pytest
import numpy as np
from ase.data.s22 import create_s22_system
from ase.build import bulk
def test_alternative_tz(factory, system):
    system.calc = factory.calc(tz=True)
    close(system.get_potential_energy(), -0.6160295884482619)