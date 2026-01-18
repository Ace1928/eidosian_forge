import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def list_system_grants_for_group(self, group_id):
    """Return a list of roles the group has on the system.

        :param group_id: the ID of the group

        :returns: a list of role assignments the group has system-wide

        """
    target_id = self._SYSTEM_SCOPE_TOKEN
    assignment_type = self._GROUP_SYSTEM
    grants = self.driver.list_system_grants(group_id, target_id, assignment_type)
    grant_ids = []
    for grant in grants:
        grant_ids.append(grant['role_id'])
    return PROVIDERS.role_api.list_roles_from_ids(grant_ids)