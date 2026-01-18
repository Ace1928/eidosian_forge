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
def do_reset_stack_status():
    print(_('Warning: this command is potentially destructive and only intended to recover from specific crashes.'))
    print(_('It is advised to shutdown all Heat engines beforehand.'))
    print(_('Continue ? [y/N]'))
    data = input()
    if not data.lower().startswith('y'):
        return
    ctxt = context.get_admin_context()
    db_api.reset_stack_status(ctxt, CONF.command.stack_id)