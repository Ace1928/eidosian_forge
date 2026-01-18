from __future__ import annotations
import dataclasses
import json
import textwrap
import os
import re
import typing as t
from . import (
from ...test import (
from ...config import (
from ...target import (
from ..integration.cloud import (
from ...io import (
from ...util import (
from ...util_common import (
from ...host_configs import (
def load_ci_config(self, python: PythonConfig) -> dict[str, t.Any]:
    """Load and return the CI YAML configuration."""
    if not self._ci_config:
        self._ci_config = self.load_yaml(python, self.CI_YML)
    return self._ci_config