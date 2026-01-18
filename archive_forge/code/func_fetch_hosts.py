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
def fetch_hosts(self):
    device_url, vm_url = self.refresh_url()
    self.devices_list = []
    self.vms_list = []
    if device_url:
        self.devices_list = self.get_resource_list(device_url)
    if vm_url:
        self.vms_list = self.get_resource_list(vm_url)
    self.devices_lookup = {device['id']: device for device in self.devices_list}
    self.vms_lookup = {vm['id']: vm for vm in self.vms_list}
    for host in self.devices_list:
        host['is_virtual'] = False
    for host in self.vms_list:
        host['is_virtual'] = True