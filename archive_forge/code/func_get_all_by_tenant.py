from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import identifier
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import resource_properties_data as rpd
@classmethod
def get_all_by_tenant(cls, context, **kwargs):
    return [cls._from_db_object(context, cls(), db_event) for db_event in db_api.event_get_all_by_tenant(context, **kwargs)]