import uuid
import fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
import requests
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
from keystoneclient import utils as base_utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import roles
from keystoneclient.v3 import users
def mock_request_method(self, request_method, body):
    return self.useFixture(fixtures.MockPatchObject(self.client, request_method, autospec=True, return_value=(self.resp, body))).mock