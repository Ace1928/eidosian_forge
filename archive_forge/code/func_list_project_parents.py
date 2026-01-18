import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_project_parents(self, project_id):
    """List all parents from a project by its ID.

        :param project_id: the driver will list the parents of this
                           project.

        :returns: a list of project_refs or an empty list.
        :raises keystone.exception.ProjectNotFound: if project_id does not
                                                    exist

        """
    raise exception.NotImplemented()