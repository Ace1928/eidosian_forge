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
def refresh_interfaces(self):
    url_device_interfaces = self.api_endpoint + '/api/dcim/interfaces/?limit=0'
    url_vm_interfaces = self.api_endpoint + '/api/virtualization/interfaces/?limit=0'
    device_interfaces = []
    vm_interfaces = []
    if self.fetch_all:
        device_interfaces = self.get_resource_list(url_device_interfaces)
        vm_interfaces = self.get_resource_list(url_vm_interfaces)
    else:
        device_interfaces = self.get_resource_list_chunked(api_url=url_device_interfaces, query_key='device_id', query_values=self.devices_lookup.keys())
        vm_interfaces = self.get_resource_list_chunked(api_url=url_vm_interfaces, query_key='virtual_machine_id', query_values=self.vms_lookup.keys())
    self.device_interfaces_lookup = defaultdict(dict)
    self.vm_interfaces_lookup = defaultdict(dict)
    self.devices_with_ips = set()
    for interface in device_interfaces:
        interface_id = interface['id']
        device_id = interface['device']['id']
        if device_id not in self.devices_lookup:
            continue
        device = self.devices_lookup[device_id]
        virtual_chassis_master = self._get_host_virtual_chassis_master(device)
        if virtual_chassis_master is not None:
            device_id = virtual_chassis_master
        self.device_interfaces_lookup[device_id][interface_id] = interface
        if interface['count_ipaddresses'] > 0:
            self.devices_with_ips.add(device_id)
    for interface in vm_interfaces:
        interface_id = interface['id']
        vm_id = interface['virtual_machine']['id']
        self.vm_interfaces_lookup[vm_id][interface_id] = interface