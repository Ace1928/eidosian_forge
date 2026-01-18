from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def collect_package_install(packages: list[str], constraints: bool=True) -> list[PipInstall]:
    """Return the details necessary to install the specified packages."""
    return collect_install([], [], packages, constraints=constraints)