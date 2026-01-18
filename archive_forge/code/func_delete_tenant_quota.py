from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def delete_tenant_quota(self, tenant, quota):
    """ deletes the tenant quotas in manageiq.

        Returns:
            result
        """
    try:
        result = self.client.post(quota['href'], action='delete')
    except Exception as e:
        self.module.fail_json(msg="failed to delete tenant quota '%s': %s" % (quota['name'], str(e)))
    return dict(changed=True, msg=result['message'])