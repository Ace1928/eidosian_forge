import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_servers_manage(self, body, **kw):
    _body = {'share_server': {'id': 'fake'}}
    resp = 202
    if not ('host' in body['share_server'] and 'share_network' in body['share_server'] and ('identifier' in body['share_server'])):
        resp = 422
    result = (resp, {}, _body)
    return result