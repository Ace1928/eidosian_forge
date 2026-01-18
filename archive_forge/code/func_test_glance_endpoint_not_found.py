from unittest import mock
from glance.async_ import utils
import glance.common.exception
from glance.tests.unit import base
def test_glance_endpoint_not_found(self):
    self.assertRaises(glance.common.exception.GlanceEndpointNotFound, utils.get_glance_endpoint, self.context, 'RegionThree', 'public')