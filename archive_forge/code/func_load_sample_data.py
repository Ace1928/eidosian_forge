import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def load_sample_data(self):
    """Create sample data to test policy associations.

        The following data is created:

        - 3 regions, in a hierarchy, 0 -> 1 -> 2 (where 0 is top)
        - 3 services
        - 6 endpoints, 2 in each region, with a mixture of services:
          0 - region 0, Service 0
          1 - region 0, Service 1
          2 - region 1, Service 1
          3 - region 1, Service 2
          4 - region 2, Service 2
          5 - region 2, Service 0

        """

    def new_endpoint(region_id, service_id):
        endpoint = unit.new_endpoint_ref(interface='test', region_id=region_id, service_id=service_id, url='/url')
        self.endpoint.append(PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint))
    self.policy = []
    self.endpoint = []
    self.service = []
    self.region = []
    parent_region_id = None
    for i in range(3):
        policy = unit.new_policy_ref()
        self.policy.append(PROVIDERS.policy_api.create_policy(policy['id'], policy))
        service = unit.new_service_ref()
        self.service.append(PROVIDERS.catalog_api.create_service(service['id'], service))
        region = unit.new_region_ref(parent_region_id=parent_region_id)
        parent_region_id = region['id']
        self.region.append(PROVIDERS.catalog_api.create_region(region))
    new_endpoint(self.region[0]['id'], self.service[0]['id'])
    new_endpoint(self.region[0]['id'], self.service[1]['id'])
    new_endpoint(self.region[1]['id'], self.service[1]['id'])
    new_endpoint(self.region[1]['id'], self.service[2]['id'])
    new_endpoint(self.region[2]['id'], self.service[2]['id'])
    new_endpoint(self.region[2]['id'], self.service[0]['id'])