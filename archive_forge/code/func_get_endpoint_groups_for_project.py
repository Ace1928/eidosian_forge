from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def get_endpoint_groups_for_project(self, project_id):
    PROVIDERS.resource_api.get_project(project_id)
    try:
        refs = self.list_endpoint_groups_for_project(project_id)
        endpoint_groups = [self.get_endpoint_group(ref['endpoint_group_id']) for ref in refs]
        return endpoint_groups
    except exception.EndpointGroupNotFound:
        return []