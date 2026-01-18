import copy
from oslo_config import cfg
from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import crypt
from heat.common import environment_format as env_fmt
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
@staticmethod
def from_db_object(context, tpl, db_tpl):
    for field in tpl.fields:
        tpl[field] = db_tpl[field]
    tpl.environment = copy.deepcopy(tpl.environment)
    if tpl.environment is not None and env_fmt.ENCRYPTED_PARAM_NAMES in tpl.environment:
        parameters = tpl.environment[env_fmt.PARAMETERS]
        encrypted_param_names = tpl.environment[env_fmt.ENCRYPTED_PARAM_NAMES]
        for param_name in encrypted_param_names:
            if isinstance(parameters[param_name], (list, tuple)) and len(parameters[param_name]) == 2:
                method, enc_value = parameters[param_name]
                value = crypt.decrypt(method, enc_value)
            else:
                value = parameters[param_name]
                LOG.warning('Encountered already-decrypted data while attempting to decrypt parameter %s. Please file a Heat bug so this can be fixed.', param_name)
            parameters[param_name] = value
        tpl.environment[env_fmt.PARAMETERS] = parameters
    tpl._context = context
    tpl.obj_reset_changes()
    return tpl