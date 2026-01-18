from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def password_change_required(self):
    """Verify whether the current password is expected array password. Works only against embedded systems."""
    if self.password is None:
        return False
    change_required = False
    system_info = None
    try:
        if self.is_proxy():
            if self.ssid == '0' or self.ssid.lower() == 'proxy':
                rc, system_info = self.request('local-users/info', force_basic_auth=False)
            elif self.is_embedded_available():
                rc, system_info = self.request('storage-systems/%s/forward/devmgr/v2/storage-systems/1/local-users/info' % self.ssid, force_basic_auth=False)
            else:
                rc, response = self.request('storage-systems/%s/passwords' % self.ssid, ignore_errors=True)
                system_info = {'minimumPasswordLength': 0, 'adminPasswordSet': response['adminPasswordSet']}
        else:
            rc, system_info = self.request('storage-systems/%s/local-users/info' % self.ssid, force_basic_auth=False)
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve information about storage system [%s]. Error [%s].' % (self.ssid, to_native(error)))
    self.is_admin_password_set = system_info['adminPasswordSet']
    if not self.is_admin_password_set:
        if self.user == 'admin' and self.password != '':
            change_required = True
    else:
        utils_login_used = False
        self.logout_system()
        if self.is_proxy():
            if self.ssid == '0' or self.ssid.lower() == 'proxy':
                utils_login_used = True
                rc, response = self.request('utils/login?uid=%s&pwd=%s&xsrf=false&onlycheck=false' % (self.user, self.password), rest_api_path=self.DEFAULT_BASE_PATH, log_request=False, ignore_errors=True, force_basic_auth=False)
            elif self.user == 'admin':
                rc, response = self.request('storage-systems/%s/stored-password/validate' % self.ssid, method='POST', log_request=False, ignore_errors=True, data={'password': self.password})
                if rc == 200:
                    change_required = not response['isValidPassword']
                elif rc == 404:
                    if self.is_web_services_version_met('04.10.0000.0000'):
                        self.module.fail_json(msg='For platforms before E2800 use SANtricity Web Services Proxy 4.1 or later! Array Id [%s].')
                    self.module.fail_json(msg='Failed to validate stored password! Array Id [%s].')
                else:
                    self.module.fail_json(msg='Failed to validate stored password! Array Id [%s].' % self.ssid)
            else:
                self.module.fail_json(msg='Role based login not available! Only storage system password can be set for storage systems prior to E2800. Array Id [%s].' % self.ssid)
        else:
            utils_login_used = True
            rc, response = self.request('utils/login?uid=%s&pwd=%s&xsrf=false&onlycheck=false' % (self.user, self.password), rest_api_path=self.DEFAULT_BASE_PATH, log_request=False, ignore_errors=True, force_basic_auth=False)
        if utils_login_used:
            if rc == 401:
                change_required = True
            elif rc == 422:
                self.module.fail_json(msg='SAML enabled! SAML disables default role based login. Array [%s]' % self.ssid)
    return change_required