import testtools
from saharaclient.api import base
from saharaclient.api import client
from keystoneauth1 import session
from requests_mock.contrib import fixture
def assertFields(self, body, obj):
    for key, value in body.items():
        self.assertEqual(value, getattr(obj, key))