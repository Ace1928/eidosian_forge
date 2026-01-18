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
def test_config_to_str_invalid_defaults():
    """Test that an error is raised if a config contains top-level keys without
    a section that would otherwise be interpreted as [DEFAULT] (which causes
    the values to be included in *all* other sections).
    """
    cfg = {'one': 1, 'two': {'@cats': 'catsie.v1', 'evil': 'hello'}}
    with pytest.raises(ConfigValidationError):
        Config(cfg).to_str()
    config_str = '[DEFAULT]\none = 1'
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)