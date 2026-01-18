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
def test_consumer_update_normalize_field(self):
    field1_name = 'some:weird-field'
    field1_orig_value = uuid.uuid4().hex
    extra_fields = {field1_name: field1_orig_value}
    consumer = self._consumer_create(**extra_fields)
    consumer_id = consumer['id']
    field1_new_value = uuid.uuid4().hex
    field2_name = 'weird:some-field'
    field2_value = uuid.uuid4().hex
    update_ref = {field1_name: field1_new_value, field2_name: field2_value}
    update_resp = self.patch(self.CONSUMER_URL + '/%s' % consumer_id, body={'consumer': update_ref})
    consumer = update_resp.result['consumer']
    normalized_field1_name = 'some_weird_field'
    self.assertEqual(field1_new_value, consumer[normalized_field1_name])
    normalized_field2_name = 'weird_some_field'
    self.assertEqual(field2_value, consumer[normalized_field2_name])