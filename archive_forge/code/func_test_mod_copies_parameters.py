import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_copies_parameters(self):
    msgid = 'Found object: %(current_value)s'
    changing_dict = {'current_value': 1}
    result = _message.Message(msgid) % changing_dict
    changing_dict['current_value'] = 2
    self.assertEqual('Found object: 1', result.translation())