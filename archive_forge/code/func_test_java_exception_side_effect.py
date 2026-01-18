import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
@unittest.skipUnless('java' in sys.platform, 'This test only applies to Jython')
def test_java_exception_side_effect(self):
    import java
    mock = Mock(side_effect=java.lang.RuntimeException('Boom!'))
    try:
        mock(1, 2, fish=3)
    except java.lang.RuntimeException:
        pass
    else:
        self.fail('java exception not raised')
    mock.assert_called_with(1, 2, fish=3)