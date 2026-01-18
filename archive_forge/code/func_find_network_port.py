from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
import os_service_types
from heat.common import config
from heat.engine.clients import client_plugin
from heat.engine import constraints
import heat.version
def find_network_port(self, value):
    return self.client().network.find_port(value).id