from __future__ import absolute_import, division, print_function
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
import datetime
import os
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def convert_tracking_params(self, module):
    body = {}
    tracking = {}
    if module.params['requester_name']:
        tracking['requesterName'] = module.params['requester_name']
    if module.params['requester_email']:
        tracking['requesterEmail'] = module.params['requester_email']
    if module.params['requester_phone']:
        tracking['requesterPhone'] = module.params['requester_phone']
    if module.params['tracking_info']:
        tracking['trackingInfo'] = module.params['tracking_info']
    if module.params['custom_fields']:
        custom_fields = {}
        for k, v in module.params['custom_fields'].items():
            if v is not None:
                custom_fields[k] = v
        tracking['customFields'] = custom_fields
    if module.params['additional_emails']:
        tracking['additionalEmails'] = module.params['additional_emails']
    body['tracking'] = tracking
    return body