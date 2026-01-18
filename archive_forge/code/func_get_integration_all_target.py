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
def get_integration_all_target(args: TestConfig) -> str:
    """Return the target to use when all tests should be run."""
    if isinstance(args, IntegrationConfig):
        return args.changed_all_target
    return 'all'