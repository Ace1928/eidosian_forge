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
def get_roles_for_groups(self, group_ids, project_id=None, domain_id=None):
    """Get a list of roles for this group on domain and/or project."""
    if not group_ids:
        return []
    if project_id is not None:
        PROVIDERS.resource_api.get_project(project_id)
        assignment_list = self.list_role_assignments(source_from_group_ids=group_ids, project_id=project_id, effective=True)
    elif domain_id is not None:
        assignment_list = self.list_role_assignments(source_from_group_ids=group_ids, domain_id=domain_id, effective=True)
    else:
        raise AttributeError(_('Must specify either domain or project'))
    role_ids = list(set([x['role_id'] for x in assignment_list]))
    return PROVIDERS.role_api.list_roles_from_ids(role_ids)