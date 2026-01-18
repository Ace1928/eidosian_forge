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
@my_registry.initializers('test_initializer.v1')
def configure_test_initializer(b: int=1) -> Callable[[int], int]:
    return partial(test_initializer, b=b)