import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_deep_copies_parameters(self):
    msgid = 'Found list: %(current_list)s'
    changing_list = list([1, 2, 3])
    params = {'current_list': changing_list}
    result = _message.Message(msgid) % params
    changing_list.append(4)
    self.assertEqual('Found list: [1, 2, 3]', result.translation())