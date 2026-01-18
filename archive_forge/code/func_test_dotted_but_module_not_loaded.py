import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import (
def test_dotted_but_module_not_loaded(self):
    import unittest.test.testmock.support
    with patch.dict('sys.modules'):
        del sys.modules['unittest.test.testmock.support']
        del sys.modules['unittest.test.testmock']
        del sys.modules['unittest.test']
        del sys.modules['unittest']

        @patch('unittest.test.testmock.support.X')
        def test(mock):
            pass
        test()