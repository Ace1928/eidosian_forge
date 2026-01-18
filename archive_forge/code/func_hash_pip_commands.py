from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
def hash_pip_commands(commands: list[PipCommand]) -> str:
    """Return a short hash unique to the given list of pip commands, suitable for identifying the resulting sanity test environment."""
    serialized_commands = json.dumps([make_pip_command_hashable(command) for command in commands], indent=4)
    return hashlib.sha256(to_bytes(serialized_commands)).hexdigest()[:8]