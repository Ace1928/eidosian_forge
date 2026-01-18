from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_igroup_rest(self):
    api = 'protocols/san/igroups'
    body = dict(name=self.parameters['name'], os_type=self.parameters['os_type'])
    body['svm'] = dict(name=self.parameters['vserver'])
    mapping = dict(initiator_group_type='protocol', bind_portset='portset', igroups='igroups', initiator_objects='initiators')
    for option in mapping:
        value = self.parameters.get(option)
        if value is not None:
            if option in ('igroups', 'initiator_objects'):
                if option == 'initiator_objects':
                    value = [self.na_helper.filter_out_none_entries(item) for item in value] if value else None
                else:
                    value = [dict(name=name) for name in value] if value else None
            if value is not None:
                body[mapping[option]] = value
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    self.fail_on_error(error)