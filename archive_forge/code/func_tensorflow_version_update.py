from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def tensorflow_version_update(module, request, response):
    auth = GcpSession(module, 'tpu')
    auth.post(''.join(['https://tpu.googleapis.com/v1/', 'projects/{project}/locations/{zone}/nodes/{name}:reimage']).format(**module.params), {u'tensorflowVersion': module.params.get('tensorflow_version')})