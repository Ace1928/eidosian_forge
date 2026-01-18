import testtools
from saharaclient.api import base
from saharaclient.api import client
from keystoneauth1 import session
from requests_mock.contrib import fixture
def assertDictsEqual(self, dict1, dict2):
    self.assertEqual(len(dict1), len(dict2))
    for key in dict1:
        self.assertEqual(dict1[key], dict2[key])