from unittest import mock
import fixtures
from urllib import parse as urlparse
import uuid
from testtools import matchers
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient import utils as client_utils
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import auth
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
def test_description_is_optional(self):
    consumer_id = uuid.uuid4().hex
    resp_ref = {'consumer': {'description': None, 'id': consumer_id}}
    self.stub_url('POST', [self.path_prefix, self.collection_key], status_code=201, json=resp_ref)
    consumer = self.manager.create()
    self.assertEqual(consumer_id, consumer.id)
    self.assertIsNone(consumer.description)