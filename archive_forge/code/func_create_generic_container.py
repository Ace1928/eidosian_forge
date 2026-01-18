from barbicanclient import exceptions
from barbicanclient.v1 import client as barbican_client
from barbicanclient.v1 import containers
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def create_generic_container(self, **props):
    return containers.Container(self.client().containers._api, **props)