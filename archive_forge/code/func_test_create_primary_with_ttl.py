import time
import uuid
from designateclient.tests import v2
def test_create_primary_with_ttl(self):
    ref = self.new_ref(email='root@example.com', ttl=60)
    self.stub_url('POST', parts=[self.RESOURCE], json=ref)
    values = ref.copy()
    del values['id']
    self.client.zones.create(values['name'], email=values['email'], ttl=values['ttl'])
    self.assertRequestBodyIs(json=values)