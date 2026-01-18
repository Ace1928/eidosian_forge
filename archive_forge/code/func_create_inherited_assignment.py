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
def create_inherited_assignment(base_ref, project_id):
    """Create a project assignment from the provided ref.

                base_ref can either be a project or domain inherited
                assignment ref.

                """
    ref = copy.deepcopy(base_ref)
    indirect = ref.setdefault('indirect', {})
    if ref.get('project_id'):
        indirect['project_id'] = ref.pop('project_id')
    else:
        indirect['domain_id'] = ref.pop('domain_id')
    ref['project_id'] = project_id
    ref.pop('inherited_to_projects')
    return ref