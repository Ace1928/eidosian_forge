from __future__ import annotations
import collections
import os
import re
import time
import typing as t
from ..target import (
from ..util import (
from .python import (
from .csharp import (
from .powershell import (
from ..config import (
from ..metadata import (
from ..data import (
def get_dependent_paths_non_recursive(self, path: str) -> list[str]:
    """Return a list of paths which depend on the given path, including dependent integration test target paths."""
    paths = self.get_dependent_paths_internal(path)
    paths += [target.path + '/' for target in self.paths_to_dependent_targets.get(path, set())]
    paths = sorted(set(paths))
    return paths