import os
import signal
import time
import fixtures
from pifpaf.drivers import rabbitmq
from oslo_messaging.tests.functional import utils
from oslo_messaging.tests import utils as test_utils
def kill_and_process(self, *args, **kargs):
    self.pifpaf.kill_node(self.n1, signal=signal.SIGKILL)
    time.sleep(0.1)
    return 'callback done'