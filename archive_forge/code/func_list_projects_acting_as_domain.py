import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_projects_acting_as_domain(self, hints):
    """List all projects acting as domains.

        :param hints: filter hints which the driver should
                      implement if at all possible.

        :returns: a list of project_refs or an empty list.

        """
    raise exception.NotImplemented()