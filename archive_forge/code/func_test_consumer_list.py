import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_consumer_list(self):
    self._consumer_create()
    resp = self.get(self.CONSUMER_URL)
    entities = resp.result['consumers']
    self.assertIsNotNone(entities)
    self_url = ['http://localhost/v3', self.CONSUMER_URL]
    self_url = ''.join(self_url)
    self.assertEqual(self_url, resp.result['links']['self'])
    self.assertValidListLinks(resp.result['links'])
    self.head(self.CONSUMER_URL, expected_status=http.client.OK)