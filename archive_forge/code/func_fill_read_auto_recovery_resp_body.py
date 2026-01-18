from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_read_auto_recovery_resp_body(body):
    result = dict()
    result['support_auto_recovery'] = body.get('support_auto_recovery')
    return result