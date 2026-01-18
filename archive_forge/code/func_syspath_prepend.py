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
def syspath_prepend(self, path) -> None:
    """Prepend ``path`` to ``sys.path`` list of import locations."""
    if self._savesyspath is None:
        self._savesyspath = sys.path[:]
    sys.path.insert(0, str(path))
    if 'pkg_resources' in sys.modules:
        from pkg_resources import fixup_namespace_packages
        fixup_namespace_packages(str(path))
    from importlib import invalidate_caches
    invalidate_caches()