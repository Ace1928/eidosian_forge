from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils import ls_helper
from gslib.utils import retry_util
from gslib.utils import text_util
from gslib.utils import unit_util
import gslib.tests.testcase as testcase
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TestParams
from gslib.utils.text_util import CompareVersions
from gslib.utils.unit_util import DecimalShort
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import PrettyTime
import httplib2
import os
import six
from six import add_move, MovedModule
from six.moves import mock
def testCompareVersions(self):
    """Tests CompareVersions for various use cases."""
    g, m = CompareVersions('3.37', '3.2')
    self.assertTrue(g)
    self.assertFalse(m)
    g, m = CompareVersions('7', '2')
    self.assertTrue(g)
    self.assertTrue(m)
    g, m = CompareVersions('3.32', '3.32pre')
    self.assertTrue(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.32pre', '3.31')
    self.assertTrue(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.4pre', '3.3pree')
    self.assertTrue(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.2', '3.37')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('2', '7')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.32pre', '3.32')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.31', '3.32pre')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.3pre', '3.3pre')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('foobar', 'baz')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.32', 'baz')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.4', '3.3')
    self.assertTrue(g)
    self.assertFalse(m)
    g, m = CompareVersions('3.3', '3.4')
    self.assertFalse(g)
    self.assertFalse(m)
    g, m = CompareVersions('4.1', '3.33')
    self.assertTrue(g)
    self.assertTrue(m)
    g, m = CompareVersions('3.10', '3.1')
    self.assertTrue(g)
    self.assertFalse(m)