from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def get_lxc_vms(self, cluster_machines, vmid=None, name=None, node=None, config=None):
    try:
        return self.get_vms_from_nodes(cluster_machines, 'lxc', vmid, name, node, config)
    except Exception as e:
        self.module.fail_json(msg='Failed to retrieve LXC VMs information: %s' % e)