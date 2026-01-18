from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def find_resource_pool_by_name(self, resource_pool_name='Resources', folder=None):
    """
        Get resource pool managed object by name
        Args:
            resource_pool_name: Name of resource pool

        Returns: Resource pool managed object if found else None

        """
    if not folder:
        folder = self.content.rootFolder
    resource_pools = get_all_objs(self.content, [vim.ResourcePool], folder=folder)
    for rp in resource_pools:
        if rp.name == resource_pool_name:
            return rp
    return None