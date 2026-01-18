import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test__map_cyclic_errors():
    import pytest
    x = np.linspace(0.2, 5.7, 10)
    pytest.raises(ValueError, _map_cyclic, x, 4.5, 3.6)
    pytest.raises(ValueError, _map_cyclic, x, 4.5, 4.5)