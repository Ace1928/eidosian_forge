import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
def test_idp_delete(self):
    idp_id = uuid.uuid4().hex
    self.client.federation.identity_providers.create(id=idp_id)
    self.client.federation.identity_providers.get(idp_id)
    self.client.federation.identity_providers.delete(idp_id)
    self.assertRaises(http.NotFound, self.client.federation.identity_providers.get, idp_id)
    idp_list = self.client.federation.identity_providers.list()
    fetched_ids = [fetched_idp.id for fetched_idp in idp_list]
    self.assertNotIn(idp_id, fetched_ids)