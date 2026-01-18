import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def remove_endpoint_group_from_project(self, endpoint_group_id, project_id):
    """Remove an endpoint to project association.

        :param endpoint_group_id: identity of endpoint to associate
        :type endpoint_group_id: string
        :param project_id: identity of project to associate
        :type project_id: string
        :raises keystone.exception.NotFound: If endpoint group project
            association was not found.
        :returns: None.

        """
    raise exception.NotImplemented()