from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
def test_duration_expired(self):
    self.assertTrue(util.Duration(0.1).expired())