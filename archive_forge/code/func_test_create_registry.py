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
def test_create_registry():
    my_registry.dogs = catalogue.create(my_registry.namespace, 'dogs', entry_points=False)
    assert hasattr(my_registry, 'dogs')
    assert len(my_registry.dogs.get_all()) == 0
    my_registry.dogs.register('good_boy.v1', func=lambda x: x)
    assert len(my_registry.dogs.get_all()) == 1