import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_projects_for_endpoint(self, endpoint_id):
    """List all projects associated with an endpoint.

        :param endpoint_id: identity of endpoint to check
        :type endpoint_id: string
        :returns: a list of projects or an empty list.

        """
    raise exception.NotImplemented()