from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_active_in_branch(self):
    script.run_script(self, '            $ brz config -d tree --scope locations --remove file\n            $ brz config -d tree file\n            branch\n            ')