from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_ntp_key(self):
    dummy, error = rest_generic.delete_async(self.rest_api, 'cluster/ntp/keys', str(self.parameters['id']))
    if error:
        self.module.fail_json(msg='Error deleting key with id %s: %s' % (self.parameters['id'], to_native(error)), exception=traceback.format_exc())