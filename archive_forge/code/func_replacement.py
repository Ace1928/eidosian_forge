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
def replacement(cls, context, existing_res_id, new_res_values, atomic_key=0, expected_engine_id=None):
    replacement = db_api.resource_create_replacement(context, existing_res_id, new_res_values, atomic_key, expected_engine_id)
    if replacement is None:
        return None
    return cls._from_db_object(cls(context), context, replacement)