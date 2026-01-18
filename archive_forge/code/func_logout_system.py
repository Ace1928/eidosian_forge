from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def logout_system(self):
    """Ensure system is logged out. This is required because login test will always succeed if previously logged in."""
    try:
        if self.is_proxy():
            if self.ssid == '0' or self.ssid.lower() == 'proxy':
                rc, system_info = self.request('utils/login', rest_api_path=self.DEFAULT_BASE_PATH, method='DELETE', force_basic_auth=False)
            elif self.is_embedded_available():
                rc, system_info = self.request('storage-systems/%s/forward/devmgr/utils/login' % self.ssid, method='DELETE', force_basic_auth=False)
            else:
                pass
        else:
            rc, system_info = self.request('utils/login', rest_api_path=self.DEFAULT_BASE_PATH, method='DELETE', force_basic_auth=False)
    except Exception as error:
        self.module.fail_json(msg='Failed to log out of storage system [%s]. Error [%s].' % (self.ssid, to_native(error)))