import numpy
import pytest
from thinc import registry
from thinc.api import (
@pytest.mark.parametrize('name,kwargs', [('glorot_uniform_init.v1', {}), ('zero_init.v1', {}), ('uniform_init.v1', {'lo': -0.5, 'hi': 0.5}), ('normal_init.v1', {'mean': 0.1})])
def test_initializer_from_config(name, kwargs):
    """Test that initializers are loaded and configured correctly from registry
    (as partials)."""
    cfg = {'test': {'@initializers': name, **kwargs}}
    func = registry.resolve(cfg)['test']
    func(NumpyOps(), (1, 2, 3, 4))