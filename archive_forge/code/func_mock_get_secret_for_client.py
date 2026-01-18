from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def mock_get_secret_for_client(client, consumers=[]):
    api_get_return = {'created': '2022-11-25T15:17:56', 'updated': '2022-11-25T15:17:56', 'status': 'ACTIVE', 'name': 'Dummy secret', 'secret_type': 'opaque', 'expiration': None, 'algorithm': None, 'bit_length': None, 'mode': None, 'creator_id': '8ddfdbc4d92440369569af0589a20fa4', 'consumers': consumers or [], 'content_types': {'default': 'text/plain'}, 'secret_ref': 'http://192.168.1.23/key-manager/v1/secrets/d46cfe10-c8ba-452f-a82f-a06834e45604'}
    client.client.get = mock.MagicMock()
    client.client.get.return_value = api_get_return