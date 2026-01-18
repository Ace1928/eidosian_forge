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
def select_and_update(self, values, expected_engine_id=None, atomic_key=0):
    return db_api.resource_update(self._context, self.id, values, atomic_key=atomic_key, expected_engine_id=expected_engine_id)