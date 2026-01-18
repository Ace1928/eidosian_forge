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
def find_network_by_name(self, network_name=None):
    """
        Get network specified by name
        Args:
            network_name: Name of network

        Returns: List of network managed objects
        """
    networks = []
    if not network_name:
        return networks
    objects = self.get_managed_objects_properties(vim_type=vim.Network, properties=['name'])
    for temp_vm_object in objects:
        if len(temp_vm_object.propSet) != 1:
            continue
        for temp_vm_object_property in temp_vm_object.propSet:
            if temp_vm_object_property.val == network_name:
                networks.append(temp_vm_object.obj)
                break
    return networks