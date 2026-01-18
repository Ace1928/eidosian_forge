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
def test_make_promise_schema():
    schema = my_registry.make_promise_schema(good_catsie)
    assert 'evil' in schema.__fields__
    assert 'cute' in schema.__fields__