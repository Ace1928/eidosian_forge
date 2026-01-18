from cinderclient.apiclient import base as common_base
from cinderclient import base
def add_project_access(self, volume_type, project):
    """Add a project to the given volume type access list."""
    info = {'project': project}
    return self._action('addProjectAccess', volume_type, info)