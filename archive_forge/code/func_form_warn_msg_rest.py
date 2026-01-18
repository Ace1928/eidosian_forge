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
def form_warn_msg_rest(self, action, code):
    start_msg = 'Quota policy rule %s opertation succeeded. ' % action
    end_msg = 'reinitialize(disable and enable again) the quota for volume %s in SVM %s.' % (self.parameters['volume'], self.parameters['vserver'])
    msg = 'unexpected code: %s' % code
    if code == '5308572':
        msg = 'However the rule is still being enforced. To stop enforcing, '
    if code in ['5308568', '5308569', '5308567']:
        msg = 'However quota resize failed due to an internal error. To make quotas active, '
    if code == '5308571':
        msg = 'but quota resize is skipped. To make quotas active, '
    self.warn_msg = start_msg + msg + end_msg