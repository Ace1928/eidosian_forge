from blazarclient import client as blazar_client
from blazarclient import exception as client_exception
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def has_host(self):
    return True if self.client().host.list() else False