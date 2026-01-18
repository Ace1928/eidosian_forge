import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def create_registered_limits(self, registered_limits):
    """Create new registered limits.

        :param registered_limits: a list of dictionaries representing limits to
                                  create.

        :returns: all the newly created registered limits.
        :raises keystone.exception.Conflict: If a duplicate registered limit
            exists.

        """
    raise exception.NotImplemented()