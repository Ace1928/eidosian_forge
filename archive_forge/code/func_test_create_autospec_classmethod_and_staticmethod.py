import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_create_autospec_classmethod_and_staticmethod(self):

    class TestClass:

        @classmethod
        def class_method(cls):
            pass

        @staticmethod
        def static_method():
            pass
    for method in ('class_method', 'static_method'):
        with self.subTest(method=method):
            mock_method = mock.create_autospec(getattr(TestClass, method))
            mock_method()
            mock_method.assert_called_once_with()
            self.assertRaises(TypeError, mock_method, 'extra_arg')