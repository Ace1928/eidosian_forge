from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def extra_statements_update(module, request, response):
    auth = GcpSession(module, 'spanner')
    auth.patch(''.join(['https://spanner.googleapis.com/v1/', 'projects/{project}/instances/{instance}/databases/{name}/ddl']).format(**module.params), {u'extraStatements': module.params.get('extra_statements')})