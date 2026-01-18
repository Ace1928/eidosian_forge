from __future__ import annotations
import os
import typing as t
from .....encoding import (
from .....data import (
from .....util_common import (
from .....executor import (
from .....provisioning import (
from ... import (
from . import (
from . import (
def is_integration_coverage_file(path: str) -> bool:
    """Returns True if the coverage file came from integration tests, otherwise False."""
    return os.path.basename(path).split('=')[0] in ('integration', 'windows-integration', 'network-integration')