import datetime
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
from keystone.models import revoke_model
from keystone.tests.unit import test_v3
def test_access_token_id_not_in_event(self):
    ref = {'description': uuid.uuid4().hex}
    resp = self.post('/OS-OAUTH1/consumers', body={'consumer': ref})
    consumer_id = resp.result['consumer']['id']
    PROVIDERS.oauth_api.delete_consumer(consumer_id)
    resp = self.get('/OS-REVOKE/events')
    events = resp.json_body['events']
    self.assertThat(events, matchers.HasLength(1))
    event = events[0]
    self.assertEqual(consumer_id, event['OS-OAUTH1:consumer_id'])
    self.assertNotIn('OS-OAUTH1:access_token_id', event)