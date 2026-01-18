import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_change_side_effect_via_delegate(self):

    def f():
        pass
    mock = create_autospec(f)
    mock.mock.side_effect = TypeError()
    with self.assertRaises(TypeError):
        mock()