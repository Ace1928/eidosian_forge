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
def test_mock_open_write(self):
    mock_namedtemp = mock.mock_open(mock.MagicMock(name='JLV'))
    with mock.patch('tempfile.NamedTemporaryFile', mock_namedtemp):
        mock_filehandle = mock_namedtemp.return_value
        mock_write = mock_filehandle.write
        mock_write.side_effect = OSError('Test 2 Error')

        def attempt():
            tempfile.NamedTemporaryFile().write('asd')
        self.assertRaises(OSError, attempt)