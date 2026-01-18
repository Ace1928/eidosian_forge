from __future__ import annotations
import os
import re
from ..io import (
from ..util import (
from .common import (
from ..data import (
from ..target import (
def get_csharp_module_utils_name(path: str) -> str:
    """Return a namespace and name from the given module_utils path."""
    base_path = data_context().content.module_utils_csharp_path
    if data_context().content.collection:
        prefix = 'ansible_collections.' + data_context().content.collection.prefix + 'plugins.module_utils.'
    else:
        prefix = ''
    name = prefix + os.path.splitext(os.path.relpath(path, base_path))[0].replace(os.path.sep, '.')
    return name