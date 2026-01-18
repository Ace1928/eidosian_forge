from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from copy import deepcopy
def diff_iscsi_config(self):
    if self.state == 'enabled':
        self.change_flag = True
        if self.existing_system_iscsi_config['iscsi_enabled'] == 'true':
            self.change_flag = False
    if self.state == 'disabled':
        self.change_flag = True
        if self.existing_system_iscsi_config['iscsi_enabled'] == 'false':
            self.change_flag = False
    if self.state == 'present':
        self.change_flag = True
        self.add_send_interface_flag = True
        if self.send_target:
            for config in self.existing_system_iscsi_config['iscsi_send_targets']:
                if config['address'] == self.send_target['address'] and config['port'] == self.send_target['port']:
                    self.change_flag = False
                    self.add_send_interface_flag = False
        self.add_static_interface_flag = True
        if self.static_target:
            for config in self.existing_system_iscsi_config['iscsi_static_targets']:
                if config['address'] == self.static_target['address'] and config['port'] == self.static_target['port'] and (config['iscsi_name'] == self.static_target['iscsi_name']):
                    self.change_flag = False
                    self.add_static_interface_flag = False
        self.update_iscsi_name_flag = False
        if self.existing_system_iscsi_config['iscsi_name'] != self.iscsi_name and self.iscsi_name:
            self.change_flag = True
            self.update_iscsi_name_flag = True
        self.update_alias_flag = False
        if self.existing_system_iscsi_config['iscsi_alias'] != self.alias:
            self.change_flag = True
            self.update_alias_flag = True
        self.update_auth_flag = False
        if self.authentication:
            auth_properties = self.existing_system_iscsi_config['iscsi_authentication_properties']
            if auth_properties['chapAuthEnabled'] != self.authentication['chap_auth_enabled']:
                self.change_flag = True
                self.update_auth_flag = True
            if auth_properties['chapAuthenticationType'] != self.authentication['chap_authentication_type']:
                self.change_flag = True
                self.update_auth_flag = True
            if auth_properties['chapName'] != self.authentication['chap_name']:
                self.change_flag = True
                self.update_auth_flag = True
            if auth_properties['mutualChapAuthenticationType'] != self.authentication['mutual_chap_authentication_type']:
                self.change_flag = True
                self.update_auth_flag = True
            if auth_properties['mutualChapName'] != self.authentication['mutual_chap_name']:
                self.change_flag = True
                self.update_auth_flag = True
        self.update_port_bind_flag = False
        if sorted(self.existing_system_iscsi_config['port_bind']) != sorted(self.port_bind):
            self.change_flag = True
            self.update_port_bind_flag = True
        self.update_send_target_authentication = False
        if self.add_send_interface_flag is False:
            for config in self.existing_system_iscsi_config['iscsi_send_targets']:
                if config['address'] == self.send_target['address'] and config['port'] == self.send_target['port']:
                    auth_properties = config['authenticationProperties']
                    if auth_properties['chapAuthEnabled'] != self.send_target['authentication']['chap_auth_enabled']:
                        self.change_flag = True
                        self.update_send_target_authentication = True
                    if auth_properties['chapAuthenticationType'] != self.send_target['authentication']['chap_authentication_type']:
                        self.change_flag = True
                        self.update_send_target_authentication = True
                    if auth_properties['chapInherited'] != self.send_target['authentication']['chap_inherited']:
                        self.change_flag = True
                        self.update_send_target_authentication = True
                    if auth_properties['chapName'] != self.send_target['authentication']['chap_name']:
                        self.change_flag = True
                        self.update_send_target_authentication = True
                    if auth_properties['mutualChapAuthenticationType'] != self.send_target['authentication']['mutual_chap_authentication_type']:
                        self.change_flag = True
                        self.update_send_target_authentication = True
                    if auth_properties['mutualChapInherited'] != self.send_target['authentication']['mutual_chap_inherited']:
                        self.change_flag = True
                        self.update_send_target_authentication = True
                    if auth_properties['mutualChapName'] != self.send_target['authentication']['mutual_chap_name']:
                        self.change_flag = True
                        self.update_send_target_authentication = True
                    break
        self.update_static_target_authentication = False
        if self.add_static_interface_flag is False:
            for config in self.existing_system_iscsi_config['iscsi_static_targets']:
                if config['address'] == self.static_target['address'] and config['port'] == self.static_target['port']:
                    auth_properties = config['authenticationProperties']
                    if auth_properties['chapAuthEnabled'] != self.static_target['authentication']['chap_auth_enabled']:
                        self.change_flag = True
                        self.update_static_target_authentication = True
                    if auth_properties['chapAuthenticationType'] != self.static_target['authentication']['chap_authentication_type']:
                        self.change_flag = True
                        self.update_static_target_authentication = True
                    if auth_properties['chapInherited'] != self.static_target['authentication']['chap_inherited']:
                        self.change_flag = True
                        self.update_static_target_authentication = True
                    if auth_properties['chapName'] != self.static_target['authentication']['chap_name']:
                        self.change_flag = True
                        self.update_static_target_authentication = True
                    if auth_properties['mutualChapAuthenticationType'] != self.static_target['authentication']['mutual_chap_authentication_type']:
                        self.change_flag = True
                        self.update_static_target_authentication = True
                    if auth_properties['mutualChapInherited'] != self.static_target['authentication']['mutual_chap_inherited']:
                        self.change_flag = True
                        self.update_static_target_authentication = True
                    if auth_properties['mutualChapName'] != self.static_target['authentication']['mutual_chap_name']:
                        self.change_flag = True
                        self.update_static_target_authentication = True
                    break
    if self.state == 'absent':
        self.change_flag = False
        self.remove_send_interface_flag = False
        if self.existing_system_iscsi_config['iscsi_send_targets'] and self.send_target:
            for config in self.existing_system_iscsi_config['iscsi_send_targets']:
                if config['address'] == self.send_target['address'] and config['port'] == self.send_target['port']:
                    self.change_flag = True
                    self.remove_send_interface_flag = True
        self.remove_static_interface_flag = False
        if self.existing_system_iscsi_config['iscsi_static_targets'] and self.static_target:
            for config in self.existing_system_iscsi_config['iscsi_static_targets']:
                if config['address'] == self.static_target['address'] and config['port'] == self.static_target['port'] and (config['iscsi_name'] == self.static_target['iscsi_name']):
                    self.change_flag = True
                    self.remove_static_interface_flag = True
        self.remove_port_bind_flag = False
        if self.iscsi_config:
            for vnic in self.port_bind:
                for existing_vnic in self.existing_system_iscsi_config['port_bind']:
                    if vnic == existing_vnic:
                        self.change_flag = True
                        self.remove_port_bind_flag = True