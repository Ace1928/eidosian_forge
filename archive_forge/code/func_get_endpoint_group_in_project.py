import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def get_endpoint_group_in_project(self, endpoint_group_id, project_id):
    """Get endpoint group to project association.

        :param endpoint_group_id: identity of endpoint group to retrieve
        :type endpoint_group_id: string
        :param project_id: identity of project to associate
        :type project_id: string
        :raises keystone.exception.NotFound: If the endpoint group to the
            project association was not found.
        :returns: a project endpoint group representation.

        """
    raise exception.NotImplemented()