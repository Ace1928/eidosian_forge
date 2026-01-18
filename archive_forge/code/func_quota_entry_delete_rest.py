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
def quota_entry_delete_rest(self):
    """
        quota_entry_delete with rest API.
        """
    if not self.use_rest:
        return self.quota_entry_delete()
    api = 'storage/quota/rules'
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.quota_uuid)
    if error is not None:
        if '5308569' in error:
            self.form_warn_msg_rest('delete', '5308569')
        elif '5308572' in error:
            self.form_warn_msg_rest('delete', '5308572')
        else:
            self.module.fail_json(msg='Error on deleting quotas rule: %s' % error)