import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_get_conf_options_undeprecated(self):
    opts = loading.get_adapter_conf_options(include_deprecated=False)
    for opt in opts:
        if opt.name.endswith('-retries'):
            self.assertIsInstance(opt, cfg.IntOpt)
        elif opt.name.endswith('-retry-delay'):
            self.assertIsInstance(opt, cfg.FloatOpt)
        elif opt.name == 'retriable-status-codes':
            self.assertIsInstance(opt, cfg.ListOpt)
        elif opt.name != 'valid-interfaces':
            self.assertIsInstance(opt, cfg.StrOpt)
        else:
            self.assertIsInstance(opt, cfg.ListOpt)
    self.assertEqual({'service-type', 'service-name', 'valid-interfaces', 'region-name', 'endpoint-override', 'version', 'min-version', 'max-version', 'connect-retries', 'status-code-retries', 'connect-retry-delay', 'status-code-retry-delay', 'retriable-status-codes'}, {opt.name for opt in opts})