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
def service_clean(self):
    ctxt = context.get_admin_context()
    for service in service_objects.Service.get_all(ctxt):
        svc = service_utils.format_service(service)
        if svc['status'] == 'down':
            service_objects.Service.delete(ctxt, svc['id'])
    print(_('Dead engines are removed.'))