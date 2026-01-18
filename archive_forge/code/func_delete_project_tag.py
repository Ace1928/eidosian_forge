from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def delete_project_tag(self, project_id, tag):
    """Delete single tag from project.

        :param project_id: The ID of the project
        :param tag: The tag value to delete

        :raises keystone.exception.ProjectTagNotFound: If the tag name
            does not exist on the project
        """
    project = self.driver.get_project(project_id)
    if ro_opt.check_resource_immutable(resource_ref=project):
        raise exception.ResourceUpdateForbidden(message=_('Cannot delete project tags for %(project_id)s, project is immutable. Set "immutable" option to false before creating project tags.') % {'project_id': project_id})
    try:
        project['tags'].remove(tag)
    except ValueError:
        raise exception.ProjectTagNotFound(project_tag=tag)
    self.update_project(project_id, project)
    notifications.Audit.deleted(self._PROJECT_TAG, tag)