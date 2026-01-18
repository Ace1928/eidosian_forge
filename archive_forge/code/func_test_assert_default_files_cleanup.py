from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def test_assert_default_files_cleanup(self):
    """Assert that using the fixture forces a clean list."""
    f = self._make_fixture()
    self.assertNotIn('default_config_files', f.conf)
    self.assertNotIn('default_config_dirs', f.conf)
    config_files = ['./test_fixture.conf']
    config_dirs = ['./test_fixture.conf.d']
    f.set_config_files(config_files)
    f.set_config_dirs(config_dirs)
    self.assertEqual(f.conf.default_config_files, config_files)
    self.assertEqual(f.conf.default_config_dirs, config_dirs)
    f.cleanUp()
    self.assertNotIn('default_config_files', f.conf)
    self.assertNotIn('default_config_dirs', f.conf)