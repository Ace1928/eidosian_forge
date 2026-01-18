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
def test_config_no_interpolation_registry():
    config_str = '[a]\nbad = true\n[b]\n@cats = "catsie.v1"\nevil = ${a:bad}\n\n[c]\n d = ${b}'
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    assert config['b']['evil'] == '${a:bad}'
    assert config['c']['d'] == '${b}'
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved['b'] == 'scratch!'
    assert resolved['c']['d'] == 'scratch!'
    assert filled['b']['evil'] == '${a:bad}'
    assert filled['b']['cute'] is True
    assert filled['c']['d'] == '${b}'
    interpolated = filled.interpolate()
    assert interpolated.is_interpolated
    assert interpolated['b']['evil'] is True
    assert interpolated['c']['d'] == interpolated['b']
    config = Config().from_str(config_str, interpolate=True)
    assert config.is_interpolated
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved['b'] == 'scratch!'
    assert resolved['c']['d'] == 'scratch!'
    assert filled['b']['evil'] is True
    assert filled['c']['d'] == filled['b']
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    filled = my_registry.fill(config)
    assert not filled.is_interpolated
    assert filled['c']['d'] == '${b}'
    resolved = my_registry.resolve(filled)
    assert resolved['c']['d'] == 'scratch!'