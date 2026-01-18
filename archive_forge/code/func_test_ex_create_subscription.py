import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_create_subscription(self):
    CloudSigmaMockHttp.type = 'CREATE_SUBSCRIPTION'
    subscription = self.driver.ex_create_subscription(amount=1, period='1 month', resource='vlan')
    self.assertEqual(subscription.amount, 1)
    self.assertEqual(subscription.period, '1 month')
    self.assertEqual(subscription.resource, 'vlan')
    self.assertEqual(subscription.price, '10.26666666666666666666666667')
    self.assertEqual(subscription.auto_renew, False)
    self.assertEqual(subscription.subscribed_object, '2494079f-8376-40bf-9b37-34d633b8a7b7')