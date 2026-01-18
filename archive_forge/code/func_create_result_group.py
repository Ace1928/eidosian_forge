from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
@staticmethod
def create_result_group(group):
    """ Creates the ansible result object from a manageiq group entity

        Returns:
            a dict with the group id, description, role, tenant, filters, group_type, created_on, updated_on
        """
    try:
        role_name = group['miq_user_role_name']
    except AttributeError:
        role_name = None
    managed_filters = None
    belongsto_filters = None
    if 'filters' in group['entitlement']:
        filters = group['entitlement']['filters']
        belongsto_filters = filters.get('belongsto')
        group_managed_filters = filters.get('managed')
        if group_managed_filters:
            managed_filters = {}
            for tag_list in group_managed_filters:
                key = tag_list[0].split('/')[2]
                tags = []
                for t in tag_list:
                    tags.append(t.split('/')[3])
                managed_filters[key] = tags
    return dict(id=group['id'], description=group['description'], role=role_name, tenant=group['tenant']['name'], managed_filters=managed_filters, belongsto_filters=belongsto_filters, group_type=group['group_type'], created_on=group['created_on'], updated_on=group['updated_on'])