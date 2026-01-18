import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.mark.parametrize('val', [np.arange(6).astype(float), np.arange(6), range(6)])
def test_array_good(props, val):
    props._setvalue('stress', val)
    assert props['stress'].shape == (6,)
    assert props['stress'].dtype == float