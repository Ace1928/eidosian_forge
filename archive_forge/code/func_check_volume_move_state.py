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
def check_volume_move_state(self, result):
    if self.use_rest:
        volume_move_status = self.na_helper.safe_get(result, ['movement', 'state'])
    else:
        volume_move_status = result.get_child_by_name('attributes-list').get_child_by_name('volume-move-info').get_child_content('state')
    if volume_move_status in ['success', 'done']:
        return False
    if volume_move_status in ['failed', 'alert', 'aborted']:
        self.module.fail_json(msg='Error moving volume %s: %s' % (self.parameters['name'], result.get_child_by_name('attributes-list').get_child_by_name('volume-move-info').get_child_content('details')))
    return True