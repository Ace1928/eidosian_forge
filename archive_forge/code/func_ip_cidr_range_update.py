from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def ip_cidr_range_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/regions/{region}/subnetworks/{name}/expandIpCidrRange']).format(**module.params), {u'ipCidrRange': module.params.get('ip_cidr_range')})