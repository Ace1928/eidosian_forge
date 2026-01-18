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
def get_dependent_paths_internal(self, path: str) -> list[str]:
    """Return a list of paths which depend on the given path."""
    ext = os.path.splitext(os.path.split(path)[1])[1]
    if is_subdir(path, data_context().content.module_utils_path):
        if ext == '.py':
            return self.get_python_module_utils_usage(path)
        if ext == '.psm1':
            return self.get_powershell_module_utils_usage(path)
        if ext == '.cs':
            return self.get_csharp_module_utils_usage(path)
    if is_subdir(path, data_context().content.integration_targets_path):
        return self.get_integration_target_usage(path)
    return []