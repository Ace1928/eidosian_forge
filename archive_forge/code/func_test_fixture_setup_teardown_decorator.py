from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.tests import utils as test_utils
def test_fixture_setup_teardown_decorator(self):
    conf = cfg.ConfigOpts()
    self.assertFalse(hasattr(conf.set_override, 'wrapped'))
    self.assertFalse(hasattr(conf.clear_override, 'wrapped'))
    fixture = conffixture.ConfFixture(conf)
    self.assertFalse(hasattr(conf.set_override, 'wrapped'))
    self.assertFalse(hasattr(conf.clear_override, 'wrapped'))
    self.useFixture(fixture)
    self.assertTrue(hasattr(conf.set_override, 'wrapped'))
    self.assertTrue(hasattr(conf.clear_override, 'wrapped'))
    fixture._teardown_decorator()
    self.assertFalse(hasattr(conf.set_override, 'wrapped'))
    self.assertFalse(hasattr(conf.clear_override, 'wrapped'))