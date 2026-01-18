import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def new_endpoint(region_id, service_id):
    endpoint = unit.new_endpoint_ref(interface='test', region_id=region_id, service_id=service_id, url='/url')
    self.endpoint.append(PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint))