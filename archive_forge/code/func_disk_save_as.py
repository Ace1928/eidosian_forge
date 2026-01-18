from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def disk_save_as(module, client, vm, disk_saveas, wait_timeout):
    if not disk_saveas.get('name'):
        module.fail_json(msg="Key 'name' is required for 'disk_saveas' option")
    image_name = disk_saveas.get('name')
    disk_id = disk_saveas.get('disk_id', 0)
    if not module.check_mode:
        if vm.STATE != VM_STATES.index('POWEROFF'):
            module.fail_json(msg="'disksaveas' option can be used only when the VM is in 'POWEROFF' state")
        try:
            client.vm.disksaveas(vm.ID, disk_id, image_name, 'OS', -1)
        except pyone.OneException as e:
            module.fail_json(msg=str(e))
        wait_for_poweroff(module, client, vm, wait_timeout)