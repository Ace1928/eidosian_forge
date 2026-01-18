import os
from eventlet.green import socket
from oslo_config import cfg
from oslo_db import options as oslo_db_ops
from oslo_log import log as logging
from oslo_middleware import cors
from oslo_policy import opts as policy_opts
from osprofiler import opts as profiler
from heat.common import exception
from heat.common.i18n import _
from heat.common import wsgi
def get_client_option(client, option):
    try:
        group_name = 'clients_' + client
        cfg.CONF.import_opt(option, 'heat.common.config', group=group_name)
        v = getattr(getattr(cfg.CONF, group_name), option)
        if v is not None:
            return v
    except cfg.NoSuchGroupError:
        pass
    cfg.CONF.import_opt(option, 'heat.common.config', group='clients')
    return getattr(cfg.CONF.clients, option)