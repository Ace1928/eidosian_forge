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
def set_by_stack_id(self, resources):
    for res in resources.values():
        self.by_stack_id_name[res.stack_id][res.name] = res