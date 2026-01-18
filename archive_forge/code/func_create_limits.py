import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def create_limits(self, limits):
    """Create new limits.

        :param limits: a list of dictionaries representing limits to create.

        :returns: all the newly created limits.
        :raises keystone.exception.Conflict: If a duplicate limit exists.
        :raises keystone.exception.NoLimitReference: If no reference registered
            limit exists.

        """
    raise exception.NotImplemented()