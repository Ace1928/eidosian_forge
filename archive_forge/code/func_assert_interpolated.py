from ase import Atoms
from ase.neb import interpolate
from ase.constraints import FixAtoms
import numpy as np
import pytest
def assert_interpolated(values):
    step = (values[-1] - values[0]) / (len(values) - 1)
    for v1, v2 in zip(*[values[i:i + 1] for i in range(len(values) - 1)]):
        assert v2 - v1 == pytest.approx(step)