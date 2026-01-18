from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def get_interface_record_rest(self, if_type, query, fields):
    if 'ipspace' in self.parameters and if_type == 'ip':
        query['ipspace.name'] = self.parameters['ipspace']
    return rest_generic.get_one_record(self.rest_api, self.get_net_int_api(if_type), query, fields)