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
def list_user_ids_for_project(self, project_id):
    PROVIDERS.resource_api.get_project(project_id)
    assignment_list = self.list_role_assignments(project_id=project_id, effective=True)
    return list(set([x['user_id'] for x in assignment_list]))