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
def test_config_deep_merge():
    config = {'a': 'hello', 'b': {'c': 'd'}}
    defaults = {'a': 'world', 'b': {'c': 'e', 'f': 'g'}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 2
    assert merged['a'] == 'hello'
    assert merged['b'] == {'c': 'd', 'f': 'g'}
    config = {'a': 'hello', 'b': {'@test': 'x', 'foo': 1}}
    defaults = {'a': 'world', 'b': {'@test': 'x', 'foo': 100, 'bar': 2}, 'c': 100}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged['a'] == 'hello'
    assert merged['b'] == {'@test': 'x', 'foo': 1, 'bar': 2}
    assert merged['c'] == 100
    config = {'a': 'hello', 'b': {'@test': 'x', 'foo': 1}, 'c': 100}
    defaults = {'a': 'world', 'b': {'@test': 'y', 'foo': 100, 'bar': 2}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged['a'] == 'hello'
    assert merged['b'] == {'@test': 'x', 'foo': 1}
    assert merged['c'] == 100
    config = {'a': 'hello', 'b': {'foo': 1}, 'c': 100}
    defaults = {'a': 'world', 'b': {'@test': 'y', 'foo': 100, 'bar': 2}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged['a'] == 'hello'
    assert merged['b'] == {'@test': 'y', 'foo': 1, 'bar': 2}
    assert merged['c'] == 100
    config = {'a': 'hello', 'b': {'@foo': 1}, 'c': 100}
    defaults = {'a': 'world', 'b': {'@bar': 'y'}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged['a'] == 'hello'
    assert merged['b'] == {'@foo': 1}
    assert merged['c'] == 100
    config = {'a': 'hello', 'b': {'@foo': 1}, 'c': 100}
    defaults = {'a': 'world', 'b': 'y'}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged['a'] == 'hello'
    assert merged['b'] == {'@foo': 1}
    assert merged['c'] == 100