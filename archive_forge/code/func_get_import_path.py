from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
def get_import_path(name: str, package: bool=False) -> str:
    """Return a path from an import name."""
    if package:
        filename = os.path.join(name.replace('.', '/'), '__init__.py')
    else:
        filename = '%s.py' % name.replace('.', '/')
    if name.startswith('ansible.module_utils.') or name == 'ansible.module_utils':
        path = os.path.join('lib', filename)
    elif data_context().content.collection and (name.startswith('ansible_collections.%s.plugins.module_utils.' % data_context().content.collection.full_name) or name == 'ansible_collections.%s.plugins.module_utils' % data_context().content.collection.full_name):
        path = '/'.join(filename.split('/')[3:])
    else:
        raise Exception('Unexpected import name: %s' % name)
    return path