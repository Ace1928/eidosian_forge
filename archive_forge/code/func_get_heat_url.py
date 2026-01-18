from oslo_config import cfg
from heatclient import client as hc
from heatclient import exc
from heat.engine.clients import client_plugin
def get_heat_url(self):
    heat_url = self._get_client_option(CLIENT_NAME, 'url')
    if heat_url:
        tenant_id = self.context.tenant_id
        heat_url = heat_url % {'tenant_id': tenant_id}
    else:
        endpoint_type = self._get_client_option(CLIENT_NAME, 'endpoint_type')
        heat_url = self.url_for(service_type=self.ORCHESTRATION, endpoint_type=endpoint_type)
    return heat_url