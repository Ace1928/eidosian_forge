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
def test_config_is_interpolated():
    """Test that a config object correctly reports whether it's interpolated."""
    config_str = '[a]\nb = 1\n\n[c]\nd = ${a:b}\ne = "hello${a:b}"\nf = ${a}'
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    config = config.merge(Config({'x': {'y': 'z'}}))
    assert not config.is_interpolated
    config = Config(config)
    assert not config.is_interpolated
    config = config.interpolate()
    assert config.is_interpolated
    config = config.merge(Config().from_str(config_str, interpolate=False))
    assert not config.is_interpolated