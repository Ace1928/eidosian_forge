from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def extract_interfaces(self, host):
    try:
        interfaces_lookup = self.vm_interfaces_lookup if host['is_virtual'] else self.device_interfaces_lookup
        interfaces = deepcopy(list(interfaces_lookup[host['id']].values()))
        before_netbox_v29 = bool(self.ipaddresses_intf_lookup)
        for interface in interfaces:
            if before_netbox_v29:
                interface['ip_addresses'] = list(self.ipaddresses_intf_lookup[interface['id']].values())
            else:
                interface['ip_addresses'] = list(self.vm_ipaddresses_intf_lookup[interface['id']].values() if host['is_virtual'] else self.device_ipaddresses_intf_lookup[interface['id']].values())
                interface['tags'] = list((sub['slug'] for sub in interface['tags']))
        return interfaces
    except Exception:
        return