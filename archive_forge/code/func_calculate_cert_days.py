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
def calculate_cert_days(expires_after):
    cert_days = 0
    if expires_after:
        expires_after_datetime = datetime.datetime.strptime(expires_after, '%Y-%m-%dT%H:%M:%SZ')
        cert_days = (expires_after_datetime - datetime.datetime.now()).days
    return cert_days