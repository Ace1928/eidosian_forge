import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_service(self, service_id):
    """Delete an existing service.

        :raises keystone.exception.ServiceNotFound: If the service doesn't
            exist.

        """
    raise exception.NotImplemented()