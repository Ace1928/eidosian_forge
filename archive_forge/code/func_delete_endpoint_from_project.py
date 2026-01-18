from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoint_groups
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import projects
def delete_endpoint_from_project(self, project, endpoint):
    """Remove a project-endpoint association."""
    if not (project and endpoint):
        raise ValueError(_('project and endpoint are required'))
    base_url = self._build_base_url(project=project, endpoint=endpoint)
    return super(EndpointFilterManager, self)._delete(url=base_url)