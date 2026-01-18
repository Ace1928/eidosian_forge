from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
from heat.objects import raw_template
from heat.objects import stack_tag
@classmethod
def get_all_by_root_owner_id(cls, context, root_owner_id):
    db_stacks = db_api.stack_get_all_by_root_owner_id(context, root_owner_id)
    for db_stack in db_stacks:
        try:
            yield cls._from_db_object(context, cls(context), db_stack)
        except exception.NotFound:
            pass