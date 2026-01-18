import datetime
import logging
import sys
import uuid
import fixtures
from kombu import connection
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import timeutils
from stevedore import dispatch
from stevedore import extension
import testscenarios
import yaml
import oslo_messaging
from oslo_messaging.notify import _impl_log
from oslo_messaging.notify import _impl_test
from oslo_messaging.notify import messaging
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_mask_passwords(self):
    driver = _impl_log.LogDriver(None, None, None)
    logger = mock.MagicMock()
    logger.info = mock.MagicMock()
    message = {'password': 'passw0rd', 'event_type': 'foo'}
    mask_str = jsonutils.dumps(strutils.mask_dict_password(message))
    with mock.patch.object(logging, 'getLogger') as gl:
        gl.return_value = logger
        driver.notify(None, message, 'info', 0)
    logger.info.assert_called_once_with(mask_str)