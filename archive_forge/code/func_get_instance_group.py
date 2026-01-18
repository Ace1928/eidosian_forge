from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def get_instance_group(self):
    if self.instance_group:
        return self.instance_group
    name = self.module.params.get('name')
    args = {'account': self.get_account('name'), 'domainid': self.get_domain('id'), 'projectid': self.get_project('id'), 'fetch_list': True}
    instance_groups = self.query_api('listInstanceGroups', **args)
    if instance_groups:
        for g in instance_groups:
            if name in [g['name'], g['id']]:
                self.instance_group = g
                break
    return self.instance_group