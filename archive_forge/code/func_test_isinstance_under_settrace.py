import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_isinstance_under_settrace(self):
    old_patch = unittest.mock.patch
    self.addCleanup(lambda patch: setattr(unittest.mock, 'patch', patch), old_patch)
    with patch.dict('sys.modules'):
        del sys.modules['unittest.mock']

        def trace(frame, event, arg):
            return trace
        self.addCleanup(sys.settrace, sys.gettrace())
        sys.settrace(trace)
        from unittest.mock import Mock, MagicMock, NonCallableMock, NonCallableMagicMock
        mocks = [Mock, MagicMock, NonCallableMock, NonCallableMagicMock, AsyncMock]
        for mock in mocks:
            obj = mock(spec=Something)
            self.assertIsInstance(obj, Something)