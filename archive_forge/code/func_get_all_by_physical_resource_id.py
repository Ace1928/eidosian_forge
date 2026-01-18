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
def get_all_by_physical_resource_id(cls, context, physical_resource_id):
    matches = db_api.resource_get_all_by_physical_resource_id(context, physical_resource_id)
    return [cls._from_db_object(cls(context), context, resource_db) for resource_db in matches]