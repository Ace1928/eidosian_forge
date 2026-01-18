import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
@pytest.fixture(scope='class')
def predictions_params(self):
    """Predictions data for eight schools."""
    return {'J': 8, 'sigma': np.array([5.0, 7.0, 12.0, 4.0, 6.0, 10.0, 3.0, 9.0])}