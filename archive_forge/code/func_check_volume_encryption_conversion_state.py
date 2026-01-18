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
def check_volume_encryption_conversion_state(self, result):
    if self.use_rest:
        volume_encryption_conversion_status = self.na_helper.safe_get(result, ['encryption', 'status', 'message'])
    else:
        volume_encryption_conversion_status = result.get_child_by_name('attributes-list').get_child_by_name('volume-encryption-conversion-info').get_child_content('status')
    if volume_encryption_conversion_status in ['running', 'initializing']:
        return True
    if volume_encryption_conversion_status in ['Not currently going on.', None]:
        return False
    self.module.fail_json(msg='Error converting encryption for volume %s: %s' % (self.parameters['name'], volume_encryption_conversion_status))