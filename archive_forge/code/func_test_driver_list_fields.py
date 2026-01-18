from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
def test_driver_list_fields(self):
    drivers = self.mgr.list(fields=['name', 'hosts'])
    expect = [('GET', '/v1/drivers/?fields=name,hosts', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(drivers, matchers.HasLength(1))