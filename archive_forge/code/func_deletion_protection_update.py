from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def deletion_protection_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://www.googleapis.com/compute/v1/', 'projects/{project}/zones/{zone}/instances/{name}/setDeletionProtection?deletionProtection={deletion_protection}']).format(**module.params), {})