from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
def list_vc_infraprofile_configs(self):
    profile_configs_list = self.api_client.appliance.infraprofile.Configs.list()
    config_list = []
    for x in profile_configs_list:
        config_list.append({'info': x.info, 'name': x.name})
    self.module.exit_json(changed=False, infra_configs_list=config_list)