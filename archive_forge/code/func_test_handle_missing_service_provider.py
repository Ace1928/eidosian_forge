import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_handle_missing_service_provider(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS)
    s = session.Session()
    auth_ref = a.get_auth_ref(s)
    service_providers = auth_ref.service_providers
    self.assertRaises(exceptions.ServiceProviderNotFound, service_providers._get_service_provider, uuid.uuid4().hex)