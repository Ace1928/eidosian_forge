from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def shielded_instance_config_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://www.googleapis.com/compute/v1/', 'projects/{project}/zones/{zone}/instances/{name}/updateShieldedInstanceConfig']).format(**module.params), {u'enableSecureBoot': navigate_hash(module.params, ['shielded_instance_config', 'enable_secure_boot']), u'enableVtpm': navigate_hash(module.params, ['shielded_instance_config', 'enable_vtpm']), u'enableIntegrityMonitoring': navigate_hash(module.params, ['shielded_instance_config', 'enable_integrity_monitoring'])})