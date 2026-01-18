from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_igroup(self, uuid, current, modify):
    for attr in ('igroups', 'initiator_names'):
        if attr in current:
            self.remove_initiators_or_igroups(uuid, attr, current[attr], current['name_to_uuid'][attr])
    for attr in ('igroups', 'initiator_names'):
        if attr in current:
            self.add_initiators_or_igroups(uuid, attr, current[attr])
        modify.pop(attr, None)
    if 'initiator_objects' in modify:
        if self.use_rest:
            changed_initiator_objects = self.change_in_initiator_comments(modify, current)
            self.modify_initiators_rest(uuid, changed_initiator_objects)
        modify.pop('initiator_objects')
    if modify:
        self.modify_igroup_rest(uuid, modify)