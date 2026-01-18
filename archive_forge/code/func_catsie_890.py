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
@my_registry.cats('catsie.v890')
def catsie_890(*args: Optional[Union[StrictBool, PositiveInt]]):
    assert args[0] == 123
    return args[0]