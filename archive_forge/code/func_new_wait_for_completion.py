import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
def new_wait_for_completion(*args, **kwargs):
    if not waited[0]:
        waited[0] = True
        complete_waiting_callback.set()
        complete_event.wait()
    old_wait_for_completion(*args, **kwargs)