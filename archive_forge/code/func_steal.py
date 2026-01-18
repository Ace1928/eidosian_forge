from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.db import api as db_api
from heat.objects import base as heat_base
@classmethod
def steal(cls, context, stack_id, old_engine_id, new_engine_id):
    return db_api.stack_lock_steal(context, stack_id, old_engine_id, new_engine_id)