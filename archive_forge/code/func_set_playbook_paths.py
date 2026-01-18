from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
def set_playbook_paths(self, playbook_paths):
    if isinstance(playbook_paths, string_types):
        playbook_paths = [playbook_paths]
    added_paths = set()
    self._n_playbook_paths = [os.path.join(to_native(p), 'collections') for p in playbook_paths if not (p in added_paths or added_paths.add(p))]
    self._n_cached_collection_paths = None
    for pkg in ['ansible_collections', 'ansible_collections.ansible']:
        self._reload_hack(pkg)