import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.mark.parametrize('val', [np.zeros(7), np.zeros((2, 3))])
def test_array_badvalue(props, val):
    with pytest.raises(ValueError):
        props._setvalue('stress', val)