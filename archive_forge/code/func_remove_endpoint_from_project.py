import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def remove_endpoint_from_project(self, endpoint_id, project_id):
    """Remove an endpoint to project association.

        :param endpoint_id: identity of endpoint to remove
        :type endpoint_id: string
        :param project_id: identity of the project associated with
        :type project_id: string
        :raises keystone.exception.NotFound: If the endpoint was not found
            in the project.
        :returns: None.

        """
    raise exception.NotImplemented()