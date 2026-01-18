import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_dict_parameter(self):
    msgid = 'Test that we can inject a dictionary %s'
    params = {'description': 'test1'}
    result = _message.Message(msgid) % params
    expected = msgid % params
    self.assertEqual(expected, result)
    self.assertEqual(expected, result.translation())