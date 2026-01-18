from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_initiators_or_igroups_rest(self, uuid, option, names):
    self.check_option_is_valid(option)
    api = 'protocols/san/igroups/%s/%s' % (uuid, self.get_rest_name_for_option(option))
    if option == 'initiator_names' and self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
        in_objects = self.parameters['initiator_objects']
        records = [self.na_helper.filter_out_none_entries(item) for item in in_objects if item['name'] in names]
    else:
        records = [dict(name=name) for name in names]
    body = dict(records=records)
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    self.fail_on_error(error)