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
def test_get_access_token_dne(self):
    url = '/users/%(user_id)s/OS-OAUTH1/access_tokens/%(key)s' % {'user_id': self.user_id, 'key': uuid.uuid4().hex}
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)