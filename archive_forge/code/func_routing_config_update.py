from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def routing_config_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.patch(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/global/networks/{name}']).format(**module.params), {u'routingConfig': NetworkRoutingconfig(module.params.get('routing_config', {}), module).to_request()})