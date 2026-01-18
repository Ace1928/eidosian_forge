from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, MutableMapping
from enum import Enum
from itertools import chain
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def clear_hosts_cache(self):
    self._hosts_cache = None
    for g in self.get_ancestors():
        g._hosts_cache = None