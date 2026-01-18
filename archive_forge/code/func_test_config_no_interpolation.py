import inspect
import pickle
import platform
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import catalogue
import pytest
from confection import Config, ConfigValidationError
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial
@pytest.mark.parametrize('d', ['.', ':'])
def test_config_no_interpolation(d):
    """Test that interpolation is correctly preserved. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    numpy = pytest.importorskip('numpy')
    c_str = f'[a]\nb = 1\n\n[c]\nd = ${{a{d}b}}\ne = "hello${{a{d}b}}"\nf = ${{a}}'
    config = Config().from_str(c_str, interpolate=False)
    assert not config.is_interpolated
    assert config['c']['d'] == f'${{a{d}b}}'
    assert config['c']['e'] == f'"hello${{a{d}b}}"'
    assert config['c']['f'] == '${a}'
    config2 = Config().from_str(config.to_str(), interpolate=True)
    assert config2.is_interpolated
    assert config2['c']['d'] == 1
    assert config2['c']['e'] == 'hello1'
    assert config2['c']['f'] == {'b': 1}
    config3 = config.interpolate()
    assert config3.is_interpolated
    assert config3['c']['d'] == 1
    assert config3['c']['e'] == 'hello1'
    assert config3['c']['f'] == {'b': 1}
    cfg = {'x': {'y': numpy.asarray([[1, 2], [4, 5]], dtype='f'), 'z': f'${{x{d}y}}'}}
    with pytest.raises(ConfigValidationError):
        Config(cfg).interpolate()