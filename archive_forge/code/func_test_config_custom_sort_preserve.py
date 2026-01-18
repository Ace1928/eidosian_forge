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
def test_config_custom_sort_preserve():
    """Test that sort order is preserved when merging and copying configs,
    or when configs are filled and resolved."""
    cfg = {'x': {}, 'y': {}, 'z': {}}
    section_order = ['y', 'z', 'x']
    expected = '[y]\n\n[z]\n\n[x]'
    config = Config(cfg, section_order=section_order)
    assert config.to_str() == expected
    config2 = config.copy()
    assert config2.to_str() == expected
    config3 = config.merge({'a': {}})
    assert config3.to_str() == f'{expected}\n\n[a]'
    config4 = Config(config)
    assert config4.to_str() == expected
    config_str = '[a]\nb = 1\n[c]\n@cats = "catsie.v1"\nevil = true\n\n[t]\n x = 2'
    section_order = ['c', 'a', 't']
    config5 = Config(section_order=section_order).from_str(config_str)
    assert list(config5.keys()) == section_order
    filled = my_registry.fill(config5)
    assert filled.section_order == section_order