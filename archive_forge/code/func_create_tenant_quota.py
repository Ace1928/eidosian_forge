from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def create_tenant_quota(self, tenant, quota_key, quota_value):
    """ Creates the tenant quotas in manageiq.

        Returns:
            result
        """
    url = '%s/quotas' % tenant['href']
    resource = {'name': quota_key, 'value': quota_value}
    try:
        self.client.post(url, action='create', resource=resource)
    except Exception as e:
        self.module.fail_json(msg='failed to create tenant quota %s: %s' % (quota_key, str(e)))
    return dict(changed=True, msg='successfully created tenant quota %s' % quota_key)