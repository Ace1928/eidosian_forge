import importlib.metadata as importlib_metadata
from unittest import mock
from heat.common import pluginutils
from heat.tests import common
def test_log_fail_msg(self):
    ep = importlib_metadata.EntryPoint(name=None, group=None, value='package.module:attr [extra1, extra2]')
    exc = Exception('Something went wrong')
    pluginutils.log_fail_msg(mock.Mock(), ep, exc)
    self.assertIn('Something went wrong', self.LOG.output)