from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_lun_rest(self):
    if self.uuid is None:
        self.module.fail_json(msg='Error deleting LUN %s: UUID not found' % self.parameters['name'])
    api = 'storage/luns'
    query = {'allow_delete_while_mapped': self.parameters['force_remove']}
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.uuid, query)
    if error:
        self.module.fail_json(msg='Error deleting LUN %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())