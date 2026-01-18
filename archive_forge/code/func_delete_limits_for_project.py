import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_limits_for_project(self, project_id):
    """Delete the existing limits which belong to the specified project.

        :param project_id: the limits' project id.

        :returns: a dictionary representing the deleted limits id. Used for
            cache invalidating.

        """
    raise exception.NotImplemented()