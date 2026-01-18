import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_none_parameter(self):
    msgid = 'Some string with params: %s'
    message = _message.Message(msgid) % None
    self.assertEqual(msgid % None, message)
    self.assertEqual(msgid % None, message.translation())