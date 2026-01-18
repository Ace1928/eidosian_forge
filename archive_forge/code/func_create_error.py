import threading
import uuid
import fixtures
import testscenarios
from oslo_messaging._drivers import pool
from oslo_messaging.tests import utils as test_utils
def create_error(retry=None):
    raise RuntimeError