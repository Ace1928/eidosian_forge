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
def set_tracking_id_by_serial_number(self, module):
    try:
        serial_number = '{0:X}'.format(self.cert.serial_number)
        cert_results = self.ecs_client.GetCertificates(serialNumber=serial_number).get('certificates', {})
        if len(cert_results) == 1:
            self.tracking_id = cert_results[0].get('trackingId')
    except RestOperationException as dummy:
        return