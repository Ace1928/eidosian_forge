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
def rename_lun_rest(self, new_path):
    if self.uuid is None:
        self.module.fail_json(msg='Error renaming LUN %s: UUID not found' % self.parameters['name'])
    api = 'storage/luns'
    body = {'name': new_path}
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
    if error:
        self.module.fail_json(msg='Error renaming LUN %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())