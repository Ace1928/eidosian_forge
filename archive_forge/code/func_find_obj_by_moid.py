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
def find_obj_by_moid(self, object_type, moid):
    """
        Get Managed Object based on an object type and moid.
        If you'd like to search for a virtual machine, recommended you use get_vm method.

        Args:
          - object_type: Managed Object type
                It is possible to specify types the following.
                ["Datacenter", "ClusterComputeResource", "ResourcePool", "Folder", "HostSystem",
                 "VirtualMachine", "DistributedVirtualSwitch", "DistributedVirtualPortgroup", "Datastore"]
          - moid: moid of Managed Object
        :return: Managed Object if it exists else None
        """
    obj = VmomiSupport.templateOf(object_type)(moid, self.si._stub)
    try:
        getattr(obj, 'name')
    except vmodl.fault.ManagedObjectNotFound:
        obj = None
    return obj