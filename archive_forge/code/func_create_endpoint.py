import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def create_endpoint(self, endpoint_id, endpoint_ref):
    """Create a new endpoint for a service.

        :raises keystone.exception.Conflict: If a duplicate endpoint exists.
        :raises keystone.exception.ServiceNotFound: If the service doesn't
            exist.

        """
    raise exception.NotImplemented()