from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
def test_backoff_delay(self):
    for _ in range(100):
        delay = util.retry_backoff_delay(self.attempt, self.scale_factor, self.jitter_max)
        self.assertThat(delay, matchers.GreaterThan(self.delay_from))
        self.assertThat(delay, matchers.LessThan(self.delay_to))