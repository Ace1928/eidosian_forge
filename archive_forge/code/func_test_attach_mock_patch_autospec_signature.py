import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_attach_mock_patch_autospec_signature(self):
    with mock.patch(f'{__name__}.Something.meth', autospec=True) as mocked:
        manager = Mock()
        manager.attach_mock(mocked, 'attach_meth')
        obj = Something()
        obj.meth(1, 2, 3, d=4)
        manager.assert_has_calls([call.attach_meth(mock.ANY, 1, 2, 3, d=4)])
        obj.meth.assert_has_calls([call(mock.ANY, 1, 2, 3, d=4)])
        mocked.assert_has_calls([call(mock.ANY, 1, 2, 3, d=4)])
    with mock.patch(f'{__name__}.something', autospec=True) as mocked:
        manager = Mock()
        manager.attach_mock(mocked, 'attach_func')
        something(1)
        manager.assert_has_calls([call.attach_func(1)])
        something.assert_has_calls([call(1)])
        mocked.assert_has_calls([call(1)])
    with mock.patch(f'{__name__}.Something', autospec=True) as mocked:
        manager = Mock()
        manager.attach_mock(mocked, 'attach_obj')
        obj = Something()
        obj.meth(1, 2, 3, d=4)
        manager.assert_has_calls([call.attach_obj(), call.attach_obj().meth(1, 2, 3, d=4)])
        obj.meth.assert_has_calls([call(1, 2, 3, d=4)])
        mocked.assert_has_calls([call(), call().meth(1, 2, 3, d=4)])