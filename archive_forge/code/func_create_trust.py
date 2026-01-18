import abc
from keystone import exception
@abc.abstractmethod
def create_trust(self, trust_id, trust, roles):
    """Create a new trust.

        :returns: a new trust
        """
    raise exception.NotImplemented()