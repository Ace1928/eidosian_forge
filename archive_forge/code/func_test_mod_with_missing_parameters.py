import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_missing_parameters(self):
    msgid = 'Some string with params: %s %s'
    test_me = lambda: _message.Message(msgid) % 'just one'
    self.assertRaises(TypeError, test_me)