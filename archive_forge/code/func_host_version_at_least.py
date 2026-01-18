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
def host_version_at_least(self, version=None, vm_obj=None, host_name=None):
    """
        Check that the ESXi Host is at least a specific version number
        Args:
            vm_obj: virtual machine object, required one of vm_obj, host_name
            host_name (string): ESXi host name
            version (tuple): a version tuple, for example (6, 7, 0)
        Returns: bool
        """
    if vm_obj:
        host_system = vm_obj.summary.runtime.host
    elif host_name:
        host_system = self.find_hostsystem_by_name(host_name=host_name)
    else:
        self.module.fail_json(msg='VM object or ESXi host name must be set one.')
    if host_system and version:
        host_version = host_system.summary.config.product.version
        return StrictVersion(host_version) >= StrictVersion('.'.join(map(str, version)))
    else:
        self.module.fail_json(msg='Unable to get the ESXi host from vm: %s, or hostname %s,or the passed ESXi version: %s is None.' % (vm_obj, host_name, version))