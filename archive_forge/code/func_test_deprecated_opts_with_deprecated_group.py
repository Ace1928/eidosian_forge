import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
def test_deprecated_opts_with_deprecated_group(self):
    results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name=None, group_obj=None, opt_list=[cfg.StrOpt('opt_name', deprecated_name='deprecated_name', deprecated_group='deprecated_group')])))
    self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: DEFAULT\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n\n          .. list-table:: Deprecated Variations\n             :header-rows: 1\n\n             - * Group\n               * Name\n             - * deprecated_group\n               * deprecated_name\n        ').lstrip(), results)