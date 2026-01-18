from unittest import mock
import requests
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import client
from cinderclient.v3 import volumes
def create_response_obj_with_header():
    resp = requests.Response()
    resp.headers['x-openstack-request-id'] = REQUEST_ID
    resp.headers['Etag'] = 'd5103bf7b26ff0310200d110da3ed186'
    resp.status_code = 200
    return resp