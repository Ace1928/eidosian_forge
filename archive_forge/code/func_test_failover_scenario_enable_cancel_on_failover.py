import os
import signal
import time
import fixtures
from pifpaf.drivers import rabbitmq
from oslo_messaging.tests.functional import utils
from oslo_messaging.tests import utils as test_utils
def test_failover_scenario_enable_cancel_on_failover(self):
    self._test_failover_scenario(enable_cancel_on_failover=True)