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
def test_config_to_str_creates_intermediate_blocks():
    cfg = Config({'optimizer': {'foo': {'bar': 1}}})
    assert cfg.to_str().strip() == '\n[optimizer]\n\n[optimizer.foo]\nbar = 1\n    '.strip()