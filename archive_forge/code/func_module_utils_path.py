from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
@property
def module_utils_path(self) -> t.Optional[str]:
    """Return the path where module_utils are found, if any."""
    return self.plugin_paths.get('module_utils')