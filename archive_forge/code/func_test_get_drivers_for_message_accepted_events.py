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
def test_get_drivers_for_message_accepted_events(self):
    config = '\ngroup_1:\n   rpc:\n       accepted_events:\n          - foo.*\n          - blah.zoo.*\n          - zip\n        '
    groups = yaml.safe_load(config)
    group = groups['group_1']
    self.assertEqual([], self.router._get_drivers_for_message(group, 'unknown', 'info'))
    self.assertEqual(['rpc'], self.router._get_drivers_for_message(group, 'foo.1', 'info'))
    self.assertEqual([], self.router._get_drivers_for_message(group, 'foo', 'info'))
    self.assertEqual(['rpc'], self.router._get_drivers_for_message(group, 'blah.zoo.zing', 'info'))