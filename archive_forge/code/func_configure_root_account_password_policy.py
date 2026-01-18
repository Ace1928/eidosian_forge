from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def configure_root_account_password_policy(self):
    default_config = self.api_client.appliance.LocalAccounts.UpdateConfig()
    current_vcenter_info = self.api_client.appliance.LocalAccounts.get('root').to_dict()
    if self._state and self.module.params['min_days_between_password_change'] > self.module.params['max_days_between_password_change']:
        self.module.fail_json('min_days_between_password_change cannot be higher than max_days_between_password_change')
    if self._state:
        _password_expiration_config = {'email': self.module.params['email'], 'min_days_between_password_change': self.module.params['min_days_between_password_change'], 'max_days_between_password_change': self.module.params['max_days_between_password_change'], 'warn_days_before_password_expiration': self.module.params['warn_days_before_password_expiration']}
    else:
        _password_expiration_config = {'max_days_between_password_change': -1}
    _changes_dict = dict()
    for k, v in _password_expiration_config.items():
        try:
            if current_vcenter_info[k] != v:
                _changes_dict[k] = v
            if k == 'fullname':
                setattr(default_config, 'full_name', v)
                continue
        except KeyError:
            "\n                Handles the case of newly installed vCenter when email field isn't present in the current config,\n                because it was never set befores\n                "
            _changes_dict[k] = v
        setattr(default_config, k, v)
    _change_result_key = 'values_would_be_changed'
    if _changes_dict:
        if not self.module.check_mode:
            _change_result_key = 'values_changed'
            self.api_client.appliance.LocalAccounts.update('root', default_config)
        self.module.exit_json(changed=True, result={_change_result_key: _changes_dict})
    self.module.exit_json(changed=False, result='No configuration changes needed')