import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_wrong_field_type(self):
    msgid = 'Test that we handle unused args %(arg1)d'
    params = {'arg1': 'test1'}
    with testtools.ExpectedException(TypeError):
        _message.Message(msgid) % params