import logging
import fixtures
import oslo_messaging
from oslo_messaging.notify import log_handler
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def fake_notifier(*args, **kwargs):
    self.stub_flg = False