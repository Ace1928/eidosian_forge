from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from copy import deepcopy
def generate_iscsi_config(self):
    self.authentication_config = ''
    self.authentication_send_target_config = ''
    self.send_target_configs = []
    self.static_target_configs = []
    if self.authentication:
        self.authentication_config = vim.host.InternetScsiHba.AuthenticationProperties()
        self.authentication_config.chapAuthEnabled = self.authentication['chap_auth_enabled']
        self.authentication_config.chapAuthenticationType = self.authentication['chap_authentication_type']
        self.authentication_config.chapName = self.authentication['chap_name']
        self.authentication_config.chapSecret = self.authentication['chap_secret']
        self.authentication_config.mutualChapAuthenticationType = self.authentication['mutual_chap_authentication_type']
        self.authentication_config.mutualChapName = self.authentication['mutual_chap_name']
        self.authentication_config.mutualChapSecret = self.authentication['mutual_chap_secret']
    if self.send_target:
        send_target_config = vim.host.InternetScsiHba.SendTarget()
        send_target_config.address = self.send_target['address']
        send_target_config.port = self.send_target['port']
        if self.send_target['authentication']:
            self.send_target_authentication_config = vim.host.InternetScsiHba.AuthenticationProperties()
            self.send_target_authentication_config.chapAuthEnabled = self.send_target['authentication']['chap_auth_enabled']
            self.send_target_authentication_config.chapAuthenticationType = self.send_target['authentication']['chap_authentication_type']
            self.send_target_authentication_config.chapInherited = self.send_target['authentication']['chap_inherited']
            self.send_target_authentication_config.chapName = self.send_target['authentication']['chap_name']
            self.send_target_authentication_config.chapSecret = self.send_target['authentication']['chap_secret']
            self.send_target_authentication_config.mutualChapAuthenticationType = self.send_target['authentication']['mutual_chap_authentication_type']
            self.send_target_authentication_config.mutualChapInherited = self.send_target['authentication']['mutual_chap_inherited']
            self.send_target_authentication_config.mutualChapName = self.send_target['authentication']['mutual_chap_name']
            self.send_target_authentication_config.mutualChapSecret = self.send_target['authentication']['mutual_chap_secret']
        self.send_target_configs.append(send_target_config)
    if self.static_target:
        static_target_config = vim.host.InternetScsiHba.StaticTarget()
        static_target_config.iScsiName = self.static_target['iscsi_name']
        static_target_config.address = self.static_target['address']
        static_target_config.port = self.static_target['port']
        if self.static_target['authentication']:
            self.static_target_authentication_config = vim.host.InternetScsiHba.AuthenticationProperties()
            self.static_target_authentication_config.chapAuthEnabled = self.static_target['authentication']['chap_auth_enabled']
            self.static_target_authentication_config.chapAuthenticationType = self.static_target['authentication']['chap_authentication_type']
            self.static_target_authentication_config.chapInherited = self.static_target['authentication']['chap_inherited']
            self.static_target_authentication_config.chapName = self.static_target['authentication']['chap_name']
            self.static_target_authentication_config.chapSecret = self.static_target['authentication']['chap_secret']
            self.static_target_authentication_config.mutualChapAuthenticationType = self.static_target['authentication']['mutual_chap_authentication_type']
            self.static_target_authentication_config.mutualChapInherited = self.static_target['authentication']['mutual_chap_inherited']
            self.static_target_authentication_config.mutualChapName = self.static_target['authentication']['mutual_chap_name']
            self.static_target_authentication_config.mutualChapSecret = self.static_target['authentication']['mutual_chap_secret']
        self.static_target_configs.append(static_target_config)