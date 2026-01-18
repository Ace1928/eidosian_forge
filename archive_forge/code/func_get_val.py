from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import exception
from heat.db import api as db_api
from heat.objects import base as heat_base
@classmethod
def get_val(cls, resource, key):
    return db_api.resource_data_get(resource.context, resource.id, key)