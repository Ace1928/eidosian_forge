import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_domain_assignments(self, domain_id):
    """Delete all assignments for a domain."""
    raise exception.NotImplemented()