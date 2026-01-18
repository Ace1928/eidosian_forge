from contextlib import contextmanager
import os
import re
import sys
from typing import Any
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
from typing import Union
import warnings
from _pytest.fixtures import fixture
from _pytest.warning_types import PytestWarning
def setenv(self, name: str, value: str, prepend: Optional[str]=None) -> None:
    """Set environment variable ``name`` to ``value``.

        If ``prepend`` is a character, read the current environment variable
        value and prepend the ``value`` adjoined with the ``prepend``
        character.
        """
    if not isinstance(value, str):
        warnings.warn(PytestWarning(f'Value of environment variable {name} type should be str, but got {value!r} (type: {type(value).__name__}); converted to str implicitly'), stacklevel=2)
        value = str(value)
    if prepend and name in os.environ:
        value = value + prepend + os.environ[name]
    self.setitem(os.environ, name, value)