from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import delay
from heat.engine import status
from heat.tests import common
from heat.tests import utils
from oslo_utils import fixture as utils_fixture
from oslo_utils import timeutils
def test_incomplete_moderate_delay(self):
    now = timeutils.utcnow()
    self.time_fixture.advance_time_seconds(2)
    poll_del = self.assertRaises(resource.PollDelay, delay.Delay._check_complete, now, 6)
    self.assertEqual(2, poll_del.period)