import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_endpoints(self, hints):
    """List all endpoints.

        :param hints: contains the list of filters yet to be satisfied.
                      Any filters satisfied here will be removed so that
                      the caller will know if any filters remain.

        :returns: list of endpoint_refs or an empty list.

        """
    raise exception.NotImplemented()