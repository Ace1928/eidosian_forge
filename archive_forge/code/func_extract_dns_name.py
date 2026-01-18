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
def extract_dns_name(self, host):
    if not host.get('primary_ip'):
        return None
    before_netbox_v29 = bool(self.ipaddresses_lookup)
    if before_netbox_v29:
        ip_address = self.ipaddresses_lookup.get(host['primary_ip']['id'])
    elif host['is_virtual']:
        ip_address = self.vm_ipaddresses_lookup.get(host['primary_ip']['id'])
    else:
        ip_address = self.device_ipaddresses_lookup.get(host['primary_ip']['id'])
    if ip_address.get('dns_name') == '':
        return None
    return ip_address.get('dns_name')