import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_projects_from_ids(self, project_ids):
    """List projects for the provided list of ids.

        :param project_ids: list of ids

        :returns: a list of project_refs.

        This method is used internally by the assignment manager to bulk read
        a set of projects given their ids.

        """
    raise exception.NotImplemented()