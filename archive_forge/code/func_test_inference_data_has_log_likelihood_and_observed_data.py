import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
@pytest.mark.skipif(packaging.version.parse(pyro.__version__) < packaging.version.parse('1.0.0'), reason='requires pyro 1.0.0 or higher')
def test_inference_data_has_log_likelihood_and_observed_data(self, data):
    idata = from_pyro(data.obj)
    test_dict = {'log_likelihood': ['obs'], 'observed_data': ['obs']}
    fails = check_multiple_attrs(test_dict, idata)
    assert not fails