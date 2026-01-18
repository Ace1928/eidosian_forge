import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def fake_update(url, body, response_key):
    return {'url': url, 'body': body, 'resp_key': response_key}