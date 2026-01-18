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
def test__check_resource_strict_id(self):
    decorated = proxy._check_resource(strict=True)(self.sot.method)
    self.assertRaisesRegex(ValueError, 'A Resource must be passed', decorated, self.sot, resource.Resource, 'this-is-not-a-resource')