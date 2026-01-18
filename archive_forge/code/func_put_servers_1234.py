from novaclient import api_versions
from novaclient.tests.unit import fakes
from novaclient.tests.unit.fixture_data import base
from novaclient.tests.unit.v2 import fakes as v2_fakes
def put_servers_1234(request, context):
    body = request.json()
    assert list(body) == ['server']
    fakes.assert_has_keys(body['server'], optional=['name', 'adminPass'])
    return request.body