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
def test_consumer_get_head(self):
    consumer = self._create_single_consumer()
    consumer_id = consumer['id']
    url = self.CONSUMER_URL + '/%s' % consumer_id
    resp = self.get(url)
    self_url = ['http://localhost/v3', self.CONSUMER_URL, '/', consumer_id]
    self_url = ''.join(self_url)
    self.assertEqual(self_url, resp.result['consumer']['links']['self'])
    self.assertEqual(consumer_id, resp.result['consumer']['id'])
    self.head(url, expected_status=http.client.OK)