import logging
from unittest import mock
from osc_lib import logs
from osc_lib.tests import utils
def test_log_level_from_options(self):
    opts = mock.Mock()
    opts.verbose_level = 0
    self.assertEqual(logging.ERROR, logs.log_level_from_options(opts))
    opts.verbose_level = 1
    self.assertEqual(logging.WARNING, logs.log_level_from_options(opts))
    opts.verbose_level = 2
    self.assertEqual(logging.INFO, logs.log_level_from_options(opts))
    opts.verbose_level = 3
    self.assertEqual(logging.DEBUG, logs.log_level_from_options(opts))