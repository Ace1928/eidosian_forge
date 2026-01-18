from novaclient import api_versions
from novaclient.tests.unit import fakes
from novaclient.tests.unit.fixture_data import base
from novaclient.tests.unit.v2 import fakes as v2_fakes
def post_servers_1234_remote_consoles(self, request, context):
    _body = ''
    body = request.json()
    context.status_code = 202
    assert len(body.keys()) == 1
    assert 'remote_console' in body.keys()
    assert 'protocol' in body['remote_console'].keys()
    protocol = body['remote_console']['protocol']
    _body = {'protocol': protocol, 'type': 'novnc', 'url': 'http://example.com:6080/vnc_auto.html?token=XYZ'}
    return {'remote_console': _body}