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
def test_consumer_update_bad_secret(self):
    consumer = self._create_single_consumer()
    original_id = consumer['id']
    update_ref = copy.deepcopy(consumer)
    update_ref['description'] = uuid.uuid4().hex
    update_ref['secret'] = uuid.uuid4().hex
    self.patch(self.CONSUMER_URL + '/%s' % original_id, body={'consumer': update_ref}, expected_status=http.client.BAD_REQUEST)