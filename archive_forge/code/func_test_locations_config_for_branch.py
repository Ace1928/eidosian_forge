from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_locations_config_for_branch(self):
    self.locations_config.set_user_option('hello', 'world')
    self.branch_config.set_user_option('hello', 'you')
    script.run_script(self, '            $ brz config -d tree\n            locations:\n              [.../tree]\n              hello = world\n            branch:\n              hello = you\n            ')