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
def test_config_pickle():
    config = Config({'foo': 'bar'}, section_order=['foo', 'bar', 'baz'])
    data = pickle.dumps(config)
    config_new = pickle.loads(data)
    assert config_new == {'foo': 'bar'}
    assert config_new.section_order == ['foo', 'bar', 'baz']