from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def test_load_custom_files(self):
    f = self._make_fixture()
    self.assertNotIn('default_config_files', f.conf)
    config_files = ['./oslo_config/tests/test_fixture.conf']
    f.set_config_files(config_files)
    opt1 = cfg.StrOpt('first_test_opt', default='initial_value_1')
    opt2 = cfg.StrOpt('second_test_opt', default='initial_value_2')
    f.register_opt(opt1)
    f.register_opt(opt2)
    self.assertEqual('loaded_value_1', f.conf.get('first_test_opt'))
    self.assertEqual('loaded_value_2', f.conf.get('second_test_opt'))