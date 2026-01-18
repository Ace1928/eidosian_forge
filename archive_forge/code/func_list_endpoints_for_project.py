import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_endpoints_for_project(self, project_id):
    """List all endpoints associated with a project.

        :param project_id: identity of the project to check
        :type project_id: string
        :returns: a list of identity endpoint ids or an empty list.

        """
    raise exception.NotImplemented()