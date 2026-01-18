import copy
import testtools
from testtools import matchers
from urllib import parse
from zunclient.common import utils as zun_utils
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import containers
def test_container_run(self):
    containers = self.mgr.run(**CREATE_CONTAINER1)
    expect = [('POST', '/v1/containers?run=true', {}, CREATE_CONTAINER1)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(containers)