from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
def get_profile_spec(self):
    infra = self.api_client.appliance.infraprofile.Configs
    profiles = {}
    profiles = self.params['profiles'].split(',')
    profile_spec = infra.ProfilesSpec(encryption_key='encryption_key', description='description', profiles=set(profiles))
    return profile_spec