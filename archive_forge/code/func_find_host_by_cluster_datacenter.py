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
def find_host_by_cluster_datacenter(module, content, datacenter_name, cluster_name, host_name):
    dc = find_datacenter_by_name(content, datacenter_name)
    if dc is None:
        module.fail_json(msg='Unable to find datacenter with name %s' % datacenter_name)
    cluster = find_cluster_by_name(content, cluster_name, datacenter=dc)
    if cluster is None:
        module.fail_json(msg='Unable to find cluster with name %s' % cluster_name)
    for host in cluster.host:
        if host.name == host_name:
            return (host, cluster)
    return (None, cluster)