from __future__ import absolute_import, division, print_function
import time
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def on_or_off_quota_rest(self, status, cd_action=None):
    """
        quota_entry_modify quota status with rest API.
        """
    if not self.use_rest:
        return self.on_or_off_quota(status, cd_action)
    body = {}
    body['quota.enabled'] = status == 'quota-on'
    api = 'storage/volumes'
    if not self.volume_uuid:
        self.volume_uuid = self.get_quota_status_or_volume_id_rest(get_volume=True)
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.volume_uuid, body)
    if error is not None:
        self.module.fail_json(msg='Error setting %s for %s: %s' % (status, self.parameters['volume'], to_native(error)))