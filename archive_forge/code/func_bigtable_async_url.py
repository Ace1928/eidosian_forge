from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def bigtable_async_url(module, extra_data=None):
    if extra_data is None:
        extra_data = {}
    location_name = module.params['clusters'][0]['location'].split('/')[-1]
    url = 'https://bigtableadmin.googleapis.com/v2/operations/projects/%s/instances/%s/locations/%s/operations/{op_id}' % (module.params['project'], module.params['name'], location_name)
    return url.format(**extra_data)