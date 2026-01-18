from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def create_ldap_account(self, account):
    args = {'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'accounttype': self.get_account_type(), 'networkdomain': self.module.params.get('network_domain'), 'username': self.module.params.get('username'), 'timezone': self.module.params.get('timezone'), 'roleid': self.get_role_id()}
    if not self.module.check_mode:
        res = self.query_api('ldapCreateAccount', **args)
        account = res['account']
        args = {'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'accounttype': self.get_account_type(), 'ldapdomain': self.module.params.get('ldap_domain'), 'type': self.module.params.get('ldap_type')}
        self.query_api('linkAccountToLdap', **args)
    return account