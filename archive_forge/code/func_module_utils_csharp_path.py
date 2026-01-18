from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
@property
def module_utils_csharp_path(self) -> t.Optional[str]:
    """Return the path where csharp module_utils are found, if any."""
    if self.is_ansible:
        return os.path.join(self.plugin_paths['module_utils'], 'csharp')
    return self.plugin_paths.get('module_utils')