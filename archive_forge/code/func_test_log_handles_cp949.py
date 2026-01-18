import locale
import unittest
from unittest import mock
from kivy import Config
from kivy.logger import Logger, FileHandler
import pytest
def test_log_handles_cp949(self):
    with mock.patch('locale.getpreferredencoding', return_value='cp949'):
        FileHandler.fd = None
        FileHandler.encoding = 'utf-8'
        Config.set('kivy', 'log_enable', 1)
        Config.set('kivy', 'log_level', 'trace')
        for string in ['한국어', 'Niñas and niños']:
            Logger.trace('Lang: call_fn => value=%r' % (string,))