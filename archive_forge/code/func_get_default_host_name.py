from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError
from ansible_collections.kubernetes.core.plugins.module_utils.common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
@staticmethod
def get_default_host_name(host):
    return host.replace('https://', '').replace('http://', '').replace('.', '-').replace(':', '_')