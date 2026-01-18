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
def test_config_deep_merge_variables():
    config_str = '[a]\nb= 1\nc = 2\n\n[d]\ne = ${a:b}'
    defaults_str = '[a]\nx = 100\n\n[d]\ny = 500'
    config = Config().from_str(config_str, interpolate=False)
    defaults = Config().from_str(defaults_str)
    merged = defaults.merge(config)
    assert merged['a'] == {'b': 1, 'c': 2, 'x': 100}
    assert merged['d'] == {'e': '${a:b}', 'y': 500}
    assert merged.interpolate()['d'] == {'e': 1, 'y': 500}
    config = Config().from_str('[a]\nb= 1\nc = 2')
    defaults = Config().from_str('[a]\nb = 100\nc = ${a:b}', interpolate=False)
    merged = defaults.merge(config)
    assert merged['a']['c'] == 2