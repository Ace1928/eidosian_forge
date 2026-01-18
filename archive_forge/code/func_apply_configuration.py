import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Set, Union
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic
from ._apply_pyprojecttoml import apply as _apply
def apply_configuration(dist: 'Distribution', filepath: _Path, ignore_option_errors=False) -> 'Distribution':
    """Apply the configuration from a ``pyproject.toml`` file into an existing
    distribution object.
    """
    config = read_configuration(filepath, True, ignore_option_errors, dist)
    return _apply(dist, config, filepath)