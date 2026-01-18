import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_contextual(self, translation):
    lang = mock.Mock()
    translation.return_value = lang
    trans = mock.Mock()
    trans.return_value = 'translated'
    lang.gettext = trans
    lang.ugettext = trans
    result = _message.Message._translate_msgid(('context', 'message'), domain='domain', has_contextual_form=True, has_plural_form=False)
    self.assertEqual('translated', result)
    trans.assert_called_with('context' + _message.CONTEXT_SEPARATOR + 'message')