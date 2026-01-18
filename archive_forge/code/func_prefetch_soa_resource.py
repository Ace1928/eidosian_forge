from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def prefetch_soa_resource(module):
    resource = SOAForwardable({'type': 'SOA', 'managed_zone': module.params['managed_zone'], 'name': replace_resource_dict(module.params['managed_zone'], 'dnsName'), 'project': module.params['project'], 'scopes': module.params['scopes'], 'service_account_file': module.params.get('service_account_file'), 'auth_kind': module.params['auth_kind'], 'service_account_email': module.params.get('service_account_email'), 'service_account_contents': module.params.get('service_account_contents')}, module)
    result = fetch_wrapped_resource(resource, 'dns#resourceRecordSet', 'dns#resourceRecordSetsListResponse', 'rrsets')
    if not result:
        raise ValueError('Google DNS Managed Zone %s not found' % replace_resource_dict(module.params['managed_zone'], 'name'))
    return result