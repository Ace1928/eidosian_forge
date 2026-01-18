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
def get_ssl_options(client):
    cacert = get_client_option(client, 'ca_file')
    insecure = get_client_option(client, 'insecure')
    cert = get_client_option(client, 'cert_file')
    key = get_client_option(client, 'key_file')
    if insecure:
        verify = False
    else:
        verify = cacert or True
    if cert and key:
        cert = (cert, key)
    return {'verify': verify, 'cert': cert}