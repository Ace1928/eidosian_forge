import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
def test__get_as_dict(self):
    resource_id = TESTABLE_RESOURCE['uuid']
    resource = self.manager._get_as_dict(resource_id)
    expect = [('GET', '/v1/testableresources/%s' % resource_id, {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(TESTABLE_RESOURCE, resource)