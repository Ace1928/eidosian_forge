import json
from unittest import mock
import uuid
from openstack import exceptions
from openstack.tests.unit import base
def test_method_not_supported(self):
    exc = exceptions.MethodNotSupported(self.__class__, 'list')
    expected = 'The list method is not supported for ' + 'openstack.tests.unit.test_exceptions.Test_Exception'
    self.assertEqual(expected, str(exc))