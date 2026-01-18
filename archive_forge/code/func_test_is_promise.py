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
def test_is_promise():
    assert my_registry.is_promise(good_catsie)
    assert not my_registry.is_promise({'hello': 'world'})
    assert not my_registry.is_promise(1)
    invalid = {'@complex': 'complex.v1', 'rate': 1.0, '@cats': 'catsie.v1'}
    assert my_registry.is_promise(invalid)