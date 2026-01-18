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
@pytest.mark.parametrize('cfg', ['[a]\nb = 1\nc = 2\n\n[a.c]\nd = 3', '[a]\nb = 1\n\n[a.c]\nd = 2\n\n[a.c.d]\ne = 3'])
def test_handle_error_duplicate_keys(cfg):
    """This would cause very cryptic error when interpreting config.
    (TypeError: 'X' object does not support item assignment)
    """
    with pytest.raises(ConfigValidationError):
        Config().from_str(cfg)