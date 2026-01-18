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
@pytest.mark.parametrize('section_order,expected_str,expected_keys', [([], '[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4\n\n[h]\ni = 5\n\n[j]\nk = 6', ['a', 'h', 'j']), (['j', 'h', 'a'], '[j]\nk = 6\n\n[h]\ni = 5\n\n[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4', ['j', 'h', 'a']), (['h'], '[h]\ni = 5\n\n[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4\n\n[j]\nk = 6', ['h', 'a', 'j'])])
def test_config_serialize_custom_sort(section_order, expected_str, expected_keys):
    cfg = {'j': {'k': 6}, 'a': {'b': 1, 'd': {'e': 3}, 'c': 2, 'f': {'g': 4}}, 'h': {'i': 5}}
    cfg_str = Config(cfg).to_str()
    assert Config(cfg, section_order=section_order).to_str() == expected_str
    keys = list(Config(section_order=section_order).from_str(cfg_str).keys())
    assert keys == expected_keys
    keys = list(Config(cfg, section_order=section_order).keys())
    assert keys == expected_keys