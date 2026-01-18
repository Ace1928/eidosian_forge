import collections
import logging
from unittest import mock
import fixtures
from oslotest import base
from testtools import compat
from testtools import matchers
from testtools import testcase
from taskflow import exceptions
from taskflow.tests import fixtures as taskflow_fixtures
from taskflow.tests import utils
from taskflow.utils import misc
def patchClass(self, module, name, autospec=True, attach_as=None):
    """Patches a modules class.

        This will create a class instance mock (using the provided name to
        find the class in the module) and attach a mock class the master mock
        to be cleaned up on test exit.
        """
    if autospec:
        instance_mock = mock.Mock(spec_set=getattr(module, name))
    else:
        instance_mock = mock.Mock()
    f = self.useFixture(fixtures.MockPatchObject(module, name, autospec=autospec))
    class_mock = f.mock
    class_mock.return_value = instance_mock
    if attach_as is None:
        attach_class_as = name
        attach_instance_as = name.lower()
    else:
        attach_class_as = attach_as + '_class'
        attach_instance_as = attach_as
    self.master_mock.attach_mock(class_mock, attach_class_as)
    self.master_mock.attach_mock(instance_mock, attach_instance_as)
    return (class_mock, instance_mock)