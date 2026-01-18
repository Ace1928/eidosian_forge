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
def get_project_tag(self, project_id, tag_name):
    """Return information for a single tag on a project.

        :param project_id: ID of a project to retrive a tag from
        :param tag_name: Name of a tag to return

        :raises keystone.exception.ProjectTagNotFound: If the tag name
            does not exist on the project
        :returns: The tag value
        """
    project = self.driver.get_project(project_id)
    if tag_name not in project.get('tags'):
        raise exception.ProjectTagNotFound(project_tag=tag_name)
    return tag_name