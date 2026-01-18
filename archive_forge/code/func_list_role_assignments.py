import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_role_assignments(self, role_id=None, user_id=None, group_ids=None, domain_id=None, project_ids=None, inherited_to_projects=None):
    """Return a list of role assignments for actors on targets.

        Available parameters represent values in which the returned role
        assignments attributes need to be filtered on.

        """
    raise exception.NotImplemented()