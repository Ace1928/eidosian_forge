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
def create_project_tag(self, project_id, tag, initiator=None):
    """Create a new tag on project.

        :param project_id: ID of a project to create a tag for
        :param tag: The string value of a tag to add

        :returns: The value of the created tag
        """
    project = self.driver.get_project(project_id)
    if ro_opt.check_resource_immutable(resource_ref=project):
        raise exception.ResourceUpdateForbidden(message=_('Cannot create project tags for %(project_id)s, project is immutable. Set "immutable" option to false before creating project tags.') % {'project_id': project_id})
    tag_name = tag.strip()
    project['tags'].append(tag_name)
    self.update_project(project_id, {'tags': project['tags']})
    notifications.Audit.created(self._PROJECT_TAG, tag_name, initiator)
    return tag_name