import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_multiple_mod_with_named_parameter(self):
    msgid = '%(description)s\nCommand: %(cmd)s\nExit code: %(exit_code)s\nStdout: %(stdout)r\nStderr: %(stderr)r'
    params = {'description': 'test1', 'cmd': 'test2', 'exit_code': 'test3', 'stdout': 'test4', 'stderr': 'test5'}
    first = _message.Message(msgid) % params
    expected = first % {}
    self.assertEqual(first.msgid, expected.msgid)
    self.assertEqual(first.params, expected.params)
    self.assertIsNot(expected, first)
    self.assertEqual(expected.translation(), first.translation())