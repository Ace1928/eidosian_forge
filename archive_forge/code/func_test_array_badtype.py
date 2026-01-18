import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.mark.parametrize('val', [5])
def test_array_badtype(props, val):
    with pytest.raises(TypeError):
        props._setvalue('stress', val)