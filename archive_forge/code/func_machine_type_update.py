from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def machine_type_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/zones/{zone}/instances/{name}/setMachineType']).format(**module.params), {u'machineType': machine_type_selflink(module.params.get('machine_type'), module.params)})