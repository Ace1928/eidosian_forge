import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_add_disabled(self):
    msgid = 'A message'
    test_me = lambda: _message.Message(msgid) + ' some string'
    self.assertRaises(TypeError, test_me)