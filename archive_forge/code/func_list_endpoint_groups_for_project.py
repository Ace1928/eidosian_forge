import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_endpoint_groups_for_project(self, project_id):
    """List all endpoint group to project associations for a project.

        :param project_id: identity of project to associate
        :type project_id: string
        :returns: None.

        """
    raise exception.NotImplemented()