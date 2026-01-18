import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
def test_deprecated_for_removal(self):
    results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', deprecated_for_removal=True, deprecated_reason='because I said so', deprecated_since='13.0')])))
    self.assertIn('.. warning::', results)
    self.assertIn('because I said so', results)
    self.assertIn('since 13.0', results)