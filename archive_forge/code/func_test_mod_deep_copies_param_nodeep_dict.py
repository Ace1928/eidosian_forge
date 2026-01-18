import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_deep_copies_param_nodeep_dict(self):
    msgid = 'Values: %(val1)s %(val2)s'
    params = {'val1': 1, 'val2': utils.NoDeepCopyObject(2)}
    result = _message.Message(msgid) % params
    self.assertEqual('Values: 1 2', result.translation())
    params = {'val1': 3, 'val2': utils.NoDeepCopyObject(4)}
    result = _message.Message(msgid) % params
    self.assertEqual('Values: 3 4', result.translation())