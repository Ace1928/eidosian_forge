from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def disruptive_update(self, changed):
    if self.parameters.get('firmware_type') == 'service-processor':
        if self.parameters.get('reboot_sp_after update'):
            self.reboot_sp()
        if not self.parameters['force_disruptive_update']:
            return
        current = self.firmware_image_get(self.parameters['node'])
        if self.parameters.get('state') == 'present' and current:
            if not self.module.check_mode:
                if self.sp_firmware_image_update():
                    changed = True
                firmware_update_progress = self.sp_firmware_image_update_progress_get(self.parameters['node'])
                while firmware_update_progress.get('is-in-progress') == 'true':
                    time.sleep(25)
                    firmware_update_progress = self.sp_firmware_image_update_progress_get(self.parameters['node'])
            else:
                changed = True
    elif self.parameters.get('firmware_type') == 'shelf':
        if self.parameters.get('shelf_module_fw'):
            if self.shelf_firmware_update_required():
                changed = True if self.module.check_mode else self.shelf_firmware_upgrade()
        else:
            changed = True if self.module.check_mode else self.shelf_firmware_upgrade()
    elif self.parameters.get('firmware_type') == 'acp' and self.acp_firmware_update_required():
        if not self.module.check_mode:
            self.acp_firmware_upgrade()
        changed = True
    elif self.parameters.get('firmware_type') == 'disk':
        if self.parameters.get('disk_fw'):
            if self.disk_firmware_update_required():
                changed = True if self.module.check_mode else self.disk_firmware_upgrade()
        else:
            changed = True if self.module.check_mode else self.disk_firmware_upgrade()
    self.module.exit_json(changed=changed, msg='forced update for %s' % self.parameters.get('firmware_type'))