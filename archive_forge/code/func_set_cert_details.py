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
def set_cert_details(self, module):
    try:
        self.cert_details = self.ecs_client.GetCertificate(trackingId=self.tracking_id)
        self.cert_status = self.cert_details.get('status')
        self.serial_number = self.cert_details.get('serialNumber')
        self.cert_days = calculate_cert_days(self.cert_details.get('expiresAfter'))
    except RestOperationException as e:
        module.fail_json('Failed to get details of certificate with tracking_id="{0}", Error: '.format(self.tracking_id), to_native(e.message))