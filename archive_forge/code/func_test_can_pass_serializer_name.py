from unittest import mock
import betamax
from betamax import exceptions
import testtools
from keystoneauth1.fixture import keystoneauth_betamax
from keystoneauth1.fixture import serializer
from keystoneauth1.fixture import v2 as v2Fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
def test_can_pass_serializer_name(self):
    fixture = keystoneauth_betamax.BetamaxFixture(cassette_name='fake', cassette_library_dir='keystoneauth1/tests/unit/data', serializer_name='json')
    self.assertIsNone(fixture.serializer)
    self.assertEqual('json', fixture.serializer_name)