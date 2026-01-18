import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('name', sorted(ADAPTERS_MANAGER.adapters))
def test_adapter_class_has_interface(name):
    """Check adapters have the correct interface."""
    assert isinstance(ADAPTERS_MANAGER.adapters[name], ContainerAdapterProtocol)