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
def get_quotas_with_retry(self, get_request, policy):
    return_values = None
    if policy is not None:
        get_request['query']['quota-entry'].add_new_child('policy', policy)
    try:
        result = self.server.invoke_successfully(get_request, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        if policy is None and 'Reason - 13001:success' in to_native(error):
            result = None
            return_values = self.debug_quota_get_error(error)
        else:
            self.module.fail_json(msg='Error fetching quotas info for policy %s: %s' % (policy, to_native(error)), exception=traceback.format_exc())
    return (result, return_values)