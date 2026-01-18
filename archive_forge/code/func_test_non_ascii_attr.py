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
def test_non_ascii_attr(self):
    r_dict = {'name': 'foobar', u'тест': '1234', u'тест2': u'привет мир'}
    r = base.Resource(None, r_dict)
    self.assertEqual(r.name, 'foobar')
    self.assertEqual(r.to_dict(), r_dict)