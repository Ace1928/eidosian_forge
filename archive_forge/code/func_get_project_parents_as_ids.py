from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def get_project_parents_as_ids(self, project):
    """Get the IDs from the parents from a given project.

        The project IDs are returned as a structured dictionary traversing up
        the hierarchy to the top level project. For example, considering the
        following project hierarchy::

                                    A
                                    |
                                  +-B-+
                                  |   |
                                  C   D

        If we query for project C parents, the expected return is the following
        dictionary::

            'parents': {
                B['id']: {
                    A['id']: None
                }
            }

        """
    parents_list = self.list_project_parents(project['id'])
    parents_as_ids = self._build_parents_as_ids_dict(project, {proj['id']: proj for proj in parents_list})
    return parents_as_ids