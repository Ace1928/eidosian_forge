import numpy as np
import pytest
from ase import Atoms
from ase.formula import Formula
@pytest.mark.parametrize('x', ['H2O', '10H2O', '2(CuO2(H2O)2)10', 'Cu20+H2', 'H' * 15, 'AuBC2', ''])
def test_formulas(x):
    f = Formula(x)
    y = str(f)
    assert y == x
    print(f.count(), '{:latex}'.format(f))
    a, b = divmod(f, 'H2O')
    assert a * Formula('H2O') + b == f
    assert f != 117