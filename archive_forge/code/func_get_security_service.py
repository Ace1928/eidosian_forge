from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
from manilaclient import client as manila_client
from manilaclient import exceptions
from oslo_config import cfg
def get_security_service(self, service_identity):
    return self._find_resource_by_id_or_name(service_identity, self.client().security_services.list(), 'security service')