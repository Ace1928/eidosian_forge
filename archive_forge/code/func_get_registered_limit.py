import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def get_registered_limit(self, registered_limit_id):
    """Get a registered limit.

        :param registered_limit_id: the registered limit id to get.

        :returns: a dictionary representing a registered limit reference.
        :raises keystone.exception.RegisteredLimitNotFound: If registered limit
            doesn't exist.

        """
    raise exception.NotImplemented()