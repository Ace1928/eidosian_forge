import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def get_limit(self, limit_id):
    """Get a limit.

        :param limit_id: the limit id to get.

        :returns: a dictionary representing a limit reference.
        :raises keystone.exception.LimitNotFound: If limit doesn't
            exist.

        """
    raise exception.NotImplemented()