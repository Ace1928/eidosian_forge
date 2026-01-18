import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.mark.parametrize('val', ['hello'])
def test_float_badvalue(props, val):
    with pytest.raises(ValueError):
        props._setvalue('energy', val)