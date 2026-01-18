import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _message
from oslo_i18n import log as i18n_log
from oslo_i18n.tests import fakes
@mock.patch('gettext.translation')
def test_emit_translated_message_with_named_args(self, mock_translation):
    log_message = 'A message to be logged %(arg1)s $(arg2)s'
    log_message_translation = 'Chinese msg to be logged %(arg1)s $(arg2)s'
    log_arg_1 = 'Arg1 to be logged'
    log_arg_1_translation = 'Arg1 to be logged in Chinese'
    log_arg_2 = 'Arg2 to be logged'
    log_arg_2_translation = 'Arg2 to be logged in Chinese'
    translations = {log_message: log_message_translation, log_arg_1: log_arg_1_translation, log_arg_2: log_arg_2_translation}
    translations_map = {'zh_CN': translations}
    translator = fakes.FakeTranslations.translator(translations_map)
    mock_translation.side_effect = translator
    msg = _message.Message(log_message)
    arg_1 = _message.Message(log_arg_1)
    arg_2 = _message.Message(log_arg_2)
    self.logger.info(msg, {'arg1': arg_1, 'arg2': arg_2})
    translation = log_message_translation % {'arg1': log_arg_1_translation, 'arg2': log_arg_2_translation}
    self.assertIn(translation, self.stream.getvalue())