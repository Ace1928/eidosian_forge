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
def find_resource_pool_by_cluster(self, resource_pool_name='Resources', cluster=None):
    """
        Get resource pool managed object by cluster object
        Args:
            resource_pool_name: Name of resource pool
            cluster: Managed object of cluster

        Returns: Resource pool managed object if found else None

        """
    desired_rp = None
    if not cluster:
        return desired_rp
    if resource_pool_name != 'Resources':
        resource_pools = cluster.resourcePool.resourcePool
        if resource_pools:
            for rp in resource_pools:
                if rp.name == resource_pool_name:
                    desired_rp = rp
                    break
    else:
        desired_rp = cluster.resourcePool
    return desired_rp