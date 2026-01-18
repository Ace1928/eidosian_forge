from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_iterate_timeout_bad_wait(self):
    with testtools.ExpectedException(exceptions.SDKException, 'Wait value must be an int or float value.'):
        for count in utils.iterate_timeout(1, 'test_iterate_timeout_bad_wait', wait='timeishard'):
            pass