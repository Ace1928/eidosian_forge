from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import limits as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import limits
def test_absolute_limits_reserved(self):
    obj = self.cs.limits.get(reserved=True)
    self.assert_request_id(obj, fakes.FAKE_REQUEST_ID_LIST)
    expected = [limits.AbsoluteLimit('maxTotalRAMSize', 51200), limits.AbsoluteLimit('maxServerMeta', 5)]
    if self.supports_image_meta:
        expected.append(limits.AbsoluteLimit('maxImageMeta', 5))
    if self.supports_personality:
        expected.extend([limits.AbsoluteLimit('maxPersonality', 5), limits.AbsoluteLimit('maxPersonalitySize', 10240)])
    self.assert_called('GET', '/limits?reserved=1')
    abs_limits = list(obj.absolute)
    self.assertEqual(len(abs_limits), len(expected))
    for limit in abs_limits:
        self.assertIn(limit, expected)