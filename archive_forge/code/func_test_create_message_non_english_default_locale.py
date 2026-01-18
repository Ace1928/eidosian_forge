import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('locale.getlocale')
@mock.patch('gettext.translation')
def test_create_message_non_english_default_locale(self, mock_translation, mock_locale):
    msgid = 'A message in English'
    es_translation = 'A message in Spanish'
    es_translations = {msgid: es_translation}
    translations_map = {'es': es_translations}
    translator = fakes.FakeTranslations.translator(translations_map)
    mock_translation.side_effect = translator
    mock_locale.return_value = ('es',)
    message = _message.Message(msgid)
    self.assertEqual(es_translation, message)
    self.assertEqual(es_translation, message.translation())