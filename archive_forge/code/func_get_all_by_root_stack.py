import collections
from oslo_config import cfg
from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
import tenacity
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
from heat.objects import resource_data
from heat.objects import resource_properties_data as rpd
@classmethod
def get_all_by_root_stack(cls, context, stack_id, filters, cache=False):
    resources_db = db_api.resource_get_all_by_root_stack(context, stack_id, filters)
    all = cls._resources_to_dict(context, resources_db)
    if cache:
        context.cache(ResourceCache).set_by_stack_id(all)
    return all