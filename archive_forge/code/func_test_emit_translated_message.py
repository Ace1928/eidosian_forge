import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _message
from oslo_i18n import log as i18n_log
from oslo_i18n.tests import fakes
@mock.patch('gettext.translation')
def test_emit_translated_message(self, mock_translation):
    log_message = 'A message to be logged'
    log_message_translation = 'A message to be logged in Chinese'
    translations = {log_message: log_message_translation}
    translations_map = {'zh_CN': translations}
    translator = fakes.FakeTranslations.translator(translations_map)
    mock_translation.side_effect = translator
    msg = _message.Message(log_message)
    self.logger.info(msg)
    self.assertIn(log_message_translation, self.stream.getvalue())