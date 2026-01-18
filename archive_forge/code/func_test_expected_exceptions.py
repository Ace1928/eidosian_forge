from unittest import mock
from heat.common import exception as heat_exception
from heat.engine.clients.os import monasca as client_plugin
from heat.tests import common
from heat.tests import utils
def test_expected_exceptions(self):
    self.assertEqual((heat_exception.EntityNotFound,), client_plugin.MonascaNotificationConstraint.expected_exceptions, 'MonascaNotificationConstraint expected exceptions error')