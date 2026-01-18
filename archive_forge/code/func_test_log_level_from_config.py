import logging
from unittest import mock
from osc_lib import logs
from osc_lib.tests import utils
def test_log_level_from_config(self):
    cfg = {'verbose_level': 0}
    self.assertEqual(logging.ERROR, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1}
    self.assertEqual(logging.WARNING, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 2}
    self.assertEqual(logging.INFO, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 3}
    self.assertEqual(logging.DEBUG, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1, 'log_level': 'critical'}
    self.assertEqual(logging.CRITICAL, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1, 'log_level': 'error'}
    self.assertEqual(logging.ERROR, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1, 'log_level': 'warning'}
    self.assertEqual(logging.WARNING, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1, 'log_level': 'info'}
    self.assertEqual(logging.INFO, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1, 'log_level': 'debug'}
    self.assertEqual(logging.DEBUG, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1, 'log_level': 'bogus'}
    self.assertEqual(logging.WARNING, logs.log_level_from_config(cfg))
    cfg = {'verbose_level': 1, 'log_level': 'info', 'debug': True}
    self.assertEqual(logging.DEBUG, logs.log_level_from_config(cfg))