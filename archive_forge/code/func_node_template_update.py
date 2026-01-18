from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def node_template_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/zones/{zone}/nodeGroups/{name}/setNodeTemplate']).format(**module.params), {u'nodeTemplate': replace_resource_dict(module.params.get(u'node_template', {}), 'selfLink')})