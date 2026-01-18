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
def test_config_to_str_simple_promises():
    """Test that references to function registries without arguments are
    serialized inline as dict."""
    config_str = '[section]\nsubsection = {"@registry":"value"}'
    config = Config().from_str(config_str)
    assert config['section']['subsection']['@registry'] == 'value'
    assert config.to_str() == config_str