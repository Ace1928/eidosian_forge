from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_aggr_actions(self):
    aggr_name = self.parameters.get('name')
    rename, cd_action, modify = (None, None, {})
    current = self.get_aggr()
    cd_action = self.na_helper.get_cd_action(current, self.parameters)
    if cd_action == 'create' and self.parameters.get('from_name'):
        old_aggregate = self.get_aggr(self.parameters['from_name'])
        rename = self.na_helper.is_rename_action(old_aggregate, current)
        if rename is None:
            self.module.fail_json(msg='Error renaming aggregate %s: no aggregate with from_name %s.' % (self.parameters['name'], self.parameters['from_name']))
        if rename:
            current = old_aggregate
            aggr_name = self.parameters['from_name']
            cd_action = None
    if cd_action is None and self.parameters['state'] == 'present':
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if 'encryption' in modify and (not self.use_rest):
            self.module.fail_json(msg='Error: modifying encryption is not supported with ZAPI.')
        if 'snaplock_type' in modify:
            self.module.fail_json(msg='Error: snaplock_type is not modifiable.  Cannot change to: %s.' % modify['snaplock_type'])
        if self.parameters.get('disks'):
            modify['disks_to_add'], modify['mirror_disks_to_add'] = self.get_disks_to_add(aggr_name, self.parameters['disks'], self.parameters.get('mirror_disks'))
        self.set_disk_count(current, modify)
    return (current, cd_action, rename, modify)