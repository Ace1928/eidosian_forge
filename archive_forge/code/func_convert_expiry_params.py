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
def convert_expiry_params(self, module):
    body = {}
    if module.params['cert_lifetime']:
        body['certLifetime'] = module.params['cert_lifetime']
    elif module.params['cert_expiry']:
        body['certExpiryDate'] = module.params['cert_expiry']
    elif self.request_type != 'reissue':
        gmt_now = datetime.datetime.fromtimestamp(time.mktime(time.gmtime()))
        expiry = gmt_now + datetime.timedelta(days=365)
        body['certExpiryDate'] = expiry.strftime('%Y-%m-%dT%H:%M:%S.00Z')
    return body