import copy
from unittest import mock
from keystoneauth1 import session
from oslo_utils import uuidutils
import novaclient.api_versions
import novaclient.client
import novaclient.extension
from novaclient.tests.unit import utils
import novaclient.v2.client
def test_client_get_reset_timings_v2(self):
    cs = novaclient.client.SessionClient(session=session.Session())
    self.assertEqual(0, len(cs.get_timings()))
    cs.times.append('somevalue')
    self.assertEqual(1, len(cs.get_timings()))
    self.assertEqual('somevalue', cs.get_timings()[0])
    cs.reset_timings()
    self.assertEqual(0, len(cs.get_timings()))