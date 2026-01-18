import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
@classmethod
def from_error(cls, err: 'ConfigValidationError', title: Optional[str]=None, desc: Optional[str]=None, parent: Optional[str]=None, show_config: Optional[bool]=None) -> 'ConfigValidationError':
    """Create a new ConfigValidationError based on an existing error, e.g.
        to re-raise it with different settings. If no overrides are provided,
        the values from the original error are used.

        err (ConfigValidationError): The original error.
        title (str): Overwrite error title.
        desc (str): Overwrite error description.
        parent (str): Overwrite error parent.
        show_config (bool): Overwrite whether to show config.
        RETURNS (ConfigValidationError): The new error.
        """
    return cls(config=err.config, errors=err.errors, title=title if title is not None else err.title, desc=desc if desc is not None else err.desc, parent=parent if parent is not None else err.parent, show_config=show_config if show_config is not None else err.show_config)