import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def get_project_by_name(self, project_name, domain_id):
    """Get a project by name.

        :returns: project_ref
        :raises keystone.exception.ProjectNotFound: if a project with the
                             project_name does not exist within the domain

        """
    raise exception.NotImplemented()