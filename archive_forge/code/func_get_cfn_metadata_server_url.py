from oslo_config import cfg
from heatclient import client as hc
from heatclient import exc
from heat.engine.clients import client_plugin
def get_cfn_metadata_server_url(self):
    config_url = cfg.CONF.heat_metadata_server_url
    if config_url is None:
        config_url = self.get_heat_cfn_url()
    if '/v1' not in config_url:
        config_url += '/v1'
    if config_url and config_url[-1] != '/':
        config_url += '/'
    return config_url