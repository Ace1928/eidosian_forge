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
def get_integration_target_usage(self, path: str) -> list[str]:
    """Return a list of paths which depend on the given path which is an integration target file."""
    target_name = path.split('/')[3]
    dependents = [os.path.join(data_context().content.integration_targets_path, target) + os.path.sep for target in sorted(self.integration_dependencies.get(target_name, set()))]
    return dependents