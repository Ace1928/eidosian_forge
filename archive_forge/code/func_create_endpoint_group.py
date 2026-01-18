import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def create_endpoint_group(self, endpoint_group):
    """Create an endpoint group.

        :param endpoint_group: endpoint group to create
        :type endpoint_group: dictionary
        :raises: keystone.exception.Conflict: If a duplicate endpoint group
            already exists.
        :returns: an endpoint group representation.

        """
    raise exception.NotImplemented()