from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
@staticmethod
def is_module_util_name(name: str) -> bool:
    """Return True if the given name is a module_util name for the content under test. External module_utils are ignored."""
    if data_context().content.is_ansible and name.startswith('ansible.module_utils.'):
        return True
    if data_context().content.collection and name.startswith('ansible_collections.%s.plugins.module_utils.' % data_context().content.collection.full_name):
        return True
    return False