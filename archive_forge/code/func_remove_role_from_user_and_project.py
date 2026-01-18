import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def remove_role_from_user_and_project(self, user_id, project_id, role_id):
    """Remove a role from a user within given project.

        :raises keystone.exception.RoleNotFound: If the role doesn't exist.

        """
    raise exception.NotImplemented()