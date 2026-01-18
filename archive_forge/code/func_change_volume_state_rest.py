from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def change_volume_state_rest(self):
    body = {'state': self.bool_to_online(self.parameters['is_online'])}
    dummy, error = self.volume_rest_patch(body)
    if error:
        self.module.fail_json(msg='Error changing state of volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    return (body['state'], None)