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
@my_registry.layers('test_layer.v1')
def test_layer(init: Callable[[int], int], c: int=1) -> Callable[[int], int]:
    return lambda x: x + init(c)