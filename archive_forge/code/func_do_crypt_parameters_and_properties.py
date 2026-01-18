import sys
from oslo_config import cfg
from oslo_log import log
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import messaging
from heat.common import service_utils
from heat.db import api as db_api
from heat.db import migration as db_migration
from heat.objects import service as service_objects
from heat.rpc import client as rpc_client
from heat import version
def do_crypt_parameters_and_properties():
    """Encrypt/decrypt hidden parameters and resource properties data."""
    ctxt = context.get_admin_context()
    prev_encryption_key = CONF.command.previous_encryption_key
    if CONF.command.crypt_operation == 'encrypt':
        db_api.encrypt_parameters_and_properties(ctxt, prev_encryption_key, CONF.command.verbose_update_params)
    elif CONF.command.crypt_operation == 'decrypt':
        db_api.decrypt_parameters_and_properties(ctxt, prev_encryption_key, CONF.command.verbose_update_params)