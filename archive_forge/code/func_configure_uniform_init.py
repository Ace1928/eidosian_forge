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
@my_registry.initializers('uniform_init.v1')
def configure_uniform_init(*, lo: float=-0.1, hi: float=0.1) -> Callable[[List[float]], List[float]]:
    return partial(uniform_init, lo=lo, hi=hi)