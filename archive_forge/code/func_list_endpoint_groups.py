import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_endpoint_groups(self, hints):
    """List all endpoint groups.

        :returns: None.

        """
    raise exception.NotImplemented()