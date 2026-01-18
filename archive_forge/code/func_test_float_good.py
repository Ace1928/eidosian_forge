import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.mark.parametrize('val', [4.0, 42, np.nan, '42.0'])
def test_float_good(props, val):
    props._setvalue('energy', val)