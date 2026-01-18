import pickle
import pytest
from sklearn.utils.metaestimators import available_if
def test_available_if_unbound_method():
    est = AvailableParameterEstimator()
    AvailableParameterEstimator.available_func(est)
    est = AvailableParameterEstimator(available=False)
    with pytest.raises(AttributeError, match="This 'AvailableParameterEstimator' has no attribute 'available_func'"):
        AvailableParameterEstimator.available_func(est)