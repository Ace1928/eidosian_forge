from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def edit_tenant(self, tenant, name, description):
    """ Edit a manageiq tenant.

        Returns:
            dict with `msg` and `changed`
        """
    resource = dict(name=name, description=description, use_config_for_attributes=False)
    if self.compare_tenant(tenant, name, description):
        return dict(changed=False, msg='tenant %s is not changed.' % tenant['name'], tenant=tenant['_data'])
    try:
        result = self.client.post(tenant['href'], action='edit', resource=resource)
    except Exception as e:
        self.module.fail_json(msg='failed to update tenant %s: %s' % (tenant['name'], str(e)))
    return dict(changed=True, msg='successfully updated the tenant with id %s' % tenant['id'])