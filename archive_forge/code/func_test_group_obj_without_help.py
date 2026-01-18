import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
def test_group_obj_without_help(self):
    results = '\n'.join(list(sphinxext._format_group_opts(namespace=None, group_name='group', group_obj=cfg.OptGroup('group'), opt_list=[cfg.StrOpt('opt_name')])))
    self.assertEqual(textwrap.dedent('\n        .. oslo.config:group:: group\n\n        .. oslo.config:option:: opt_name\n\n          :Type: string\n          :Default: ``<None>``\n        ').lstrip(), results)