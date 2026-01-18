import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_grant(self, role_id, user_id=None, group_id=None, domain_id=None, project_id=None, inherited_to_projects=False):
    """Delete assignments/grants.

        :raises keystone.exception.RoleAssignmentNotFound: If the role
            assignment doesn't exist.

        """
    raise exception.NotImplemented()