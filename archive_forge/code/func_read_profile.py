from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
def read_profile(self):
    with open(self.config_path, 'r') as file:
        return file.read()