from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_igroup_rest(self, uuid):
    api = 'protocols/san/igroups'
    query = {'allow_delete_while_mapped': True} if self.parameters['force_remove_initiator'] else None
    dummy, error = rest_generic.delete_async(self.rest_api, api, uuid, query=query)
    self.fail_on_error(error)