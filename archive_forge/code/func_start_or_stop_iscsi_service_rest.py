from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def start_or_stop_iscsi_service_rest(self, service_state):
    api = 'protocols/san/iscsi/services'
    enabled = True if service_state == 'started' else False
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, {'enabled': enabled})
    if error:
        self.module.fail_json(msg='Error %s iscsi service on vserver %s: %s' % (service_state[0:5] + 'ing', self.parameters['vserver'], error))