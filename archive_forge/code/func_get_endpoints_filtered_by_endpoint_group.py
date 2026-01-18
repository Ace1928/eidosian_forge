from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def get_endpoints_filtered_by_endpoint_group(self, endpoint_group_id):
    endpoints = self.list_endpoints()
    filters = self.get_endpoint_group(endpoint_group_id)['filters']
    filtered_endpoints = []
    for endpoint in endpoints:
        is_candidate = True
        for key, value in filters.items():
            if endpoint[key] != value:
                is_candidate = False
                break
        if is_candidate:
            filtered_endpoints.append(endpoint)
    return filtered_endpoints