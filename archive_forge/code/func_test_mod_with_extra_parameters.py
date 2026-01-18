import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_extra_parameters(self):
    msgid = 'Some string with params: %(param1)s %(param2)s'
    params = {'param1': 'test', 'param2': 'test2', 'param3': 'notinstring'}
    result = _message.Message(msgid) % params
    expected = msgid % params
    self.assertEqual(expected, result)
    self.assertEqual(expected, result.translation())
    self.assertEqual(params.keys(), result.params.keys())