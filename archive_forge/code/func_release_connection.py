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
def release_connection(self, connection):
    self._check_safe()
    self._locked_connections.remove(connection)
    self._available_connections.append(connection)