import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_endpoint_group_association_by_project(self, project_id):
    """Remove endpoint group to project associations.

        :param project_id: identity of the project to check
        :type project_id: string
        :returns: None

        """
    raise exception.NotImplemented()