from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_config_all(self):
    out, err = self.run_bzr(['config'])
    self.assertEqual('', out)
    self.assertEqual('', err)