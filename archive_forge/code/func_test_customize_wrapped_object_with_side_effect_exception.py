import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_customize_wrapped_object_with_side_effect_exception(self):

    class Real(object):

        def method(self):
            pass
    real = Real()
    mock = Mock(wraps=real)
    mock.method.side_effect = RuntimeError
    self.assertRaises(RuntimeError, mock.method)