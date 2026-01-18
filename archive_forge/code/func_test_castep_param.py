import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
def test_castep_param(testing_keywords):
    cparam = CastepParam(testing_keywords, keyword_tolerance=2)
    cparam.continuation = True
    with pytest.warns(UserWarning):
        cparam.reuse = False
    cparam.continuation = None
    cparam.reuse = True
    with pytest.warns(UserWarning):
        cparam.continuation = True
    cparam.cut_off_energy = 500
    with pytest.warns(UserWarning):
        cparam.basis_precision = 'FINE'