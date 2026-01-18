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
def test_config_interpolation_lists():
    c_str = '[a]\nb = 1\n\n[c]\nd = ["hello ${a.b}", "world"]'
    config = Config().from_str(c_str, interpolate=False)
    assert config['c']['d'] == ['hello ${a.b}', 'world']
    config = config.interpolate()
    assert config['c']['d'] == ['hello 1', 'world']
    c_str = '[a]\nb = 1\n\n[c]\nd = [${a.b}, "hello ${a.b}", "world"]'
    config = Config().from_str(c_str)
    assert config['c']['d'] == [1, 'hello 1', 'world']
    config = Config().from_str(c_str, interpolate=False)
    c_str = '[a]\nb = 1\n\n[c]\nd = ["hello", ${a}]'
    config = Config().from_str(c_str)
    assert config['c']['d'] == ['hello', {'b': 1}]
    c_str = '[a]\nb = 1\n\n[c]\nd = ["hello", "hello ${a}"]'
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    config_str = '[a]\nb = 1\n\n[c]\nd = ["hello", {"x": ["hello ${a.b}"], "y": 2}]'
    config = Config().from_str(config_str)
    assert config['c']['d'] == ['hello', {'x': ['hello 1'], 'y': 2}]
    config_str = '[a]\nb = 1\n\n[c]\nd = ["hello", {"x": [${a.b}], "y": 2}]'
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)