from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def get_vms_from_cluster_resources(self):
    try:
        return self.proxmox_api.cluster().resources().get(type='vm')
    except Exception as e:
        self.module.fail_json(msg='Failed to retrieve VMs information from cluster resources: %s' % e)