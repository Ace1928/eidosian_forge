from novaclient import api_versions
from novaclient.tests.unit import fakes
from novaclient.tests.unit.fixture_data import base
from novaclient.tests.unit.v2 import fakes as v2_fakes
def put_server_tag(request, context):
    assert request.text is None
    context.status_code = 201
    return None