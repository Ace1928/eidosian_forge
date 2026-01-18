from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def private_ip_google_access_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/regions/{region}/subnetworks/{name}/setPrivateIpGoogleAccess']).format(**module.params), {u'privateIpGoogleAccess': module.params.get('private_ip_google_access')})