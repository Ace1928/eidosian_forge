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
@staticmethod
def encrypt_properties_data(data):
    if cfg.CONF.encrypt_parameters_and_properties and data:
        result = crypt.encrypted_dict(data)
        return (True, result)
    return (False, data)