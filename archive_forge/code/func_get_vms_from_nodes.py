from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
def get_vms_from_nodes(self, cluster_machines, type, vmid=None, name=None, node=None, config=None):
    filtered_vms = {vm: info for vm, info in cluster_machines.items() if not (type != info['type'] or (node and info['node'] != node) or (vmid and int(info['vmid']) != vmid) or (name is not None and info['name'] != name))}
    nodes = frozenset([info['node'] for vm, info in filtered_vms.items()])
    for this_node in nodes:
        call_vm_getter = getattr(self.proxmox_api.nodes(this_node), type)
        vms_from_this_node = call_vm_getter().get()
        for detected_vm in vms_from_this_node:
            this_vm_id = int(detected_vm['vmid'])
            desired_vm = filtered_vms.get(this_vm_id, None)
            if desired_vm:
                desired_vm.update(detected_vm)
                desired_vm['vmid'] = this_vm_id
                desired_vm['template'] = proxmox_to_ansible_bool(desired_vm['template'])
                if config != 'none':
                    config_type = 0 if config == 'pending' else 1
                    desired_vm['config'] = call_vm_getter(this_vm_id).config().get(current=config_type)
    return filtered_vms