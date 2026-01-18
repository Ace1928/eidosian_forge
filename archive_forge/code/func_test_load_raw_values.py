from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def test_load_raw_values(self):
    f = self._make_fixture()
    f.load_raw_values(first_test_opt='loaded_value_1', second_test_opt='loaded_value_2')
    self.assertRaises(cfg.NoSuchOptError, f.conf.get, 'first_test_opt')
    self.assertRaises(cfg.NoSuchOptError, f.conf.get, 'second_test_opt')
    opt1 = cfg.StrOpt('first_test_opt', default='initial_value_1')
    opt2 = cfg.StrOpt('second_test_opt', default='initial_value_2')
    f.register_opt(opt1)
    f.register_opt(opt2)
    self.assertEqual(f.conf.first_test_opt, 'loaded_value_1')
    self.assertEqual(f.conf.second_test_opt, 'loaded_value_2')
    f.cleanUp()
    self.assertRaises(cfg.NoSuchOptError, f.conf.get, 'first_test_opt')
    self.assertRaises(cfg.NoSuchOptError, f.conf.get, 'second_test_opt')
    f.register_opt(opt1)
    f.register_opt(opt2)
    self.assertEqual(f.conf.first_test_opt, 'initial_value_1')
    self.assertEqual(f.conf.second_test_opt, 'initial_value_2')