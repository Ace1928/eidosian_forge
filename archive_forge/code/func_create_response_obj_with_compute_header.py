import requests
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def create_response_obj_with_compute_header():
    resp = requests.Response()
    resp.headers['x-compute-request-id'] = fakes.FAKE_REQUEST_ID
    return resp