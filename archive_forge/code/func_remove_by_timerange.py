from __future__ import (absolute_import, division, print_function)
import collections
import os
import time
from multiprocessing import Lock
from itertools import chain
from ansible.errors import AnsibleError
from ansible.module_utils.common._collections_compat import MutableSet
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
def remove_by_timerange(self, s_min, s_max):
    for k in list(self._keyset.keys()):
        t = self._keyset[k]
        if s_min < t < s_max:
            del self._keyset[k]
    self._cache.set(self.PREFIX, self._keyset)