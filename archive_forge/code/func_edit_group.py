from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def edit_group(self, group, description, role, tenant, norm_managed_filters, managed_filters_merge_mode, belongsto_filters, belongsto_filters_merge_mode):
    """ Edit a manageiq group.

        Returns:
            a dict of:
            changed: boolean indicating if the entity was updated.
            msg: a short message describing the operation executed.
        """
    if role or norm_managed_filters or belongsto_filters:
        group.reload(attributes=['miq_user_role_name', 'entitlement'])
    try:
        current_role = group['miq_user_role_name']
    except AttributeError:
        current_role = None
    changed = False
    resource = {}
    if description and group['description'] != description:
        resource['description'] = description
        changed = True
    if tenant and group['tenant_id'] != tenant['id']:
        resource['tenant'] = dict(id=tenant['id'])
        changed = True
    if role and current_role != role['name']:
        resource['role'] = dict(id=role['id'])
        changed = True
    if norm_managed_filters or belongsto_filters:
        entitlement = group['entitlement']
        if 'filters' not in entitlement:
            managed_tag_filters_post_body = self.normalized_managed_tag_filters_to_miq(norm_managed_filters)
            resource['filters'] = {'managed': managed_tag_filters_post_body, 'belongsto': belongsto_filters}
            changed = True
        else:
            current_filters = entitlement['filters']
            new_filters = self.edit_group_edit_filters(current_filters, norm_managed_filters, managed_filters_merge_mode, belongsto_filters, belongsto_filters_merge_mode)
            if new_filters:
                resource['filters'] = new_filters
                changed = True
    if not changed:
        return dict(changed=False, msg='group %s is not changed.' % group['description'])
    try:
        self.client.post(group['href'], action='edit', resource=resource)
        changed = True
    except Exception as e:
        self.module.fail_json(msg='failed to update group %s: %s' % (group['name'], str(e)))
    return dict(changed=changed, msg='successfully updated the group %s with id %s' % (group['description'], group['id']))