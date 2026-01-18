from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_list_value_all(self):
    config.option_registry.register(config.ListOption('list'))
    self.addCleanup(config.option_registry.remove, 'list')
    self.breezy_config.set_user_option('list', [1, 'a', 'with, a comma'])
    script.run_script(self, '            $ brz config -d tree\n            breezy:\n              [DEFAULT]\n              list = 1, a, "with, a comma"\n            ')