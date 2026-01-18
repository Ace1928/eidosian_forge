import pickle
import pytest
from sklearn.utils.metaestimators import available_if
def test_available_if_methods_can_be_pickled():
    """Check that available_if methods can be pickled.

    Non-regression test for #21344.
    """
    return_value = 10
    est = AvailableParameterEstimator(available=True, return_value=return_value)
    pickled_bytes = pickle.dumps(est.available_func)
    unpickled_func = pickle.loads(pickled_bytes)
    assert unpickled_func() == return_value