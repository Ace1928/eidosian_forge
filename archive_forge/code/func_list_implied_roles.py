import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_implied_roles(self, prior_role_id):
    """List roles implied from the prior role ID."""
    raise exception.NotImplemented()