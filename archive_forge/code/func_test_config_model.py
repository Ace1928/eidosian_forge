import threading
from unittest import mock
import greenlet
from oslo_config import cfg
from oslotest import base
from oslo_reports.generators import conf as os_cgen
from oslo_reports.generators import threading as os_tgen
from oslo_reports.generators import version as os_pgen
from oslo_reports.models import threading as os_tmod
def test_config_model(self):
    conf = cfg.ConfigOpts()
    conf.register_opt(cfg.StrOpt('crackers', default='triscuit'))
    conf.register_opt(cfg.StrOpt('secrets', secret=True, default='should not show'))
    conf.register_group(cfg.OptGroup('cheese', title='Cheese Info'))
    conf.register_opt(cfg.IntOpt('sharpness', default=1), group='cheese')
    conf.register_opt(cfg.StrOpt('name', default='cheddar'), group='cheese')
    conf.register_opt(cfg.BoolOpt('from_cow', default=True), group='cheese')
    conf.register_opt(cfg.StrOpt('group_secrets', secret=True, default='should not show'), group='cheese')
    model = os_cgen.ConfigReportGenerator(conf)()
    model.set_current_view_type('text')
    config_source_line = '  config_source = \n'
    try:
        conf.config_source
    except cfg.NoSuchOptError:
        config_source_line = ''
    target_str = '\ncheese: \n  from_cow = True\n  group_secrets = ***\n  name = cheddar\n  sharpness = 1\n\ndefault: \n%s  crackers = triscuit\n  secrets = ***' % config_source_line
    self.assertEqual(target_str, str(model))