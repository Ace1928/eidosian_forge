from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def tenant_quota(self, tenant, quota_key):
    """ Search for tenant quota object by tenant and quota_key.
        Returns:
            the quota for the tenant, or None if the tenant quota was not found.
        """
    tenant_quotas = self.client.get('%s/quotas?expand=resources&filter[]=name=%s' % (tenant['href'], quota_key))
    return tenant_quotas['resources']