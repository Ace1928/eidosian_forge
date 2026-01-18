from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoint_groups
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import projects
def list_projects_for_endpoint_group(self, endpoint_group):
    """List all projects associated with a given endpoint group."""
    if not endpoint_group:
        raise ValueError(_('endpoint_group is required'))
    base_url = self._build_group_base_url(endpoint_group=endpoint_group)
    return super(EndpointFilterManager, self)._list(base_url, projects.ProjectManager.collection_key, obj_class=projects.ProjectManager.resource_class)