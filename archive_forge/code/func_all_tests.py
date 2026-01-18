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
def all_tests(args: TestConfig, force: bool=False) -> dict[str, str]:
    """Return the targets for each test command when all tests should be run."""
    if force:
        integration_all_target = 'all'
    else:
        integration_all_target = get_integration_all_target(args)
    return {'sanity': 'all', 'units': 'all', 'integration': integration_all_target, 'windows-integration': integration_all_target, 'network-integration': integration_all_target}