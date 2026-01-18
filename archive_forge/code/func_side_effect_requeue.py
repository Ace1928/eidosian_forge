import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def side_effect_requeue(*args, **kwargs):
    if endpoint.info.call_count == 1:
        return oslo_messaging.NotificationResult.REQUEUE
    return oslo_messaging.NotificationResult.HANDLED