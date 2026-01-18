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
def get_target_name(path: str) -> str:
    """Extract the test target name from the given coverage path."""
    return to_text(os.path.basename(path).split('=')[1])