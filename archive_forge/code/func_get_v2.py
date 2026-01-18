import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_v2(self, **kw):
    body = {'versions': [{'status': 'CURRENT', 'updated': '2015-07-30T11:33:21Z', 'links': [{'href': 'http://docs.openstack.org/', 'type': 'text/html', 'rel': 'describedby'}, {'href': 'http://localhost:8786/v2/', 'rel': 'self'}], 'min_version': '2.0', 'version': '2.5', 'id': 'v1.0'}]}
    return (200, {}, body)