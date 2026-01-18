import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test__check_resource_incorrect_resource(self):

    class OneType(resource.Resource):
        pass

    class AnotherType(resource.Resource):
        pass
    value = AnotherType()
    decorated = proxy._check_resource(strict=False)(self.sot.method)
    self.assertRaisesRegex(ValueError, 'Expected OneType but received AnotherType', decorated, self.sot, OneType, value)