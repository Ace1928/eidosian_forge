import os
from heat.common.i18n import _
from heatclient import client as heat_client
from keystoneauth1.identity.generic import password
from keystoneauth1 import session
from keystoneclient.v3 import client as kc_v3
from novaclient import client as nova_client
from swiftclient import client as swift_client
def get_endpoint_url(self, service_type, region=None):
    kwargs = {'service_type': service_type, 'region_name': region}
    return self.auth_ref.service_catalog.url_for(**kwargs)