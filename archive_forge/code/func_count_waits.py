import threading
import uuid
import fixtures
import testscenarios
from oslo_messaging._drivers import pool
from oslo_messaging.tests import utils as test_utils
def count_waits(**kwargs):
    self.n_waits += 1
    self.orig_wait(**kwargs)