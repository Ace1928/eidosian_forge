import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_list_subscriptions(self):
    subscriptions = self.driver.ex_list_subscriptions()
    self.assertEqual(len(subscriptions), 6)
    subscription = subscriptions[0]
    self.assertEqual(subscription.id, '7272')
    self.assertEqual(subscription.resource, 'vlan')
    self.assertEqual(subscription.amount, 1)
    self.assertEqual(subscription.period, '345 days, 0:00:00')
    self.assertEqual(subscription.status, 'active')
    self.assertEqual(subscription.price, '0E-20')
    subscription = subscriptions[-1]
    self.assertEqual(subscription.id, '5555')
    self.assertEqual(subscription.start_time, None)
    self.assertEqual(subscription.end_time, None)