from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def minimum_password_length_change_required(self):
    """Retrieve the current storage array's global configuration."""
    change_required = False
    try:
        if self.is_proxy():
            if self.ssid == '0' or self.ssid.lower() == 'proxy':
                rc, system_info = self.request('local-users/info', force_basic_auth=False)
            elif self.is_embedded_available():
                rc, system_info = self.request('storage-systems/%s/forward/devmgr/v2/storage-systems/1/local-users/info' % self.ssid, force_basic_auth=False)
            else:
                return False
        else:
            rc, system_info = self.request('storage-systems/%s/local-users/info' % self.ssid, force_basic_auth=False)
    except Exception as error:
        self.module.fail_json(msg='Failed to determine minimum password length. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    self.is_admin_password_set = system_info['adminPasswordSet']
    if self.minimum_password_length is not None and self.minimum_password_length != system_info['minimumPasswordLength']:
        change_required = True
    if self.password is not None and (change_required and self.minimum_password_length > len(self.password) or (not change_required and system_info['minimumPasswordLength'] > len(self.password))):
        self.module.fail_json(msg='Password does not meet the length requirement [%s]. Array Id [%s].' % (system_info['minimumPasswordLength'], self.ssid))
    return change_required