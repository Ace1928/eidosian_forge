from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_branch_config_default(self):
    script.run_script(self, '            $ brz config -d tree --scope locations --remove file\n            $ brz config -d tree --all file\n            branch:\n              file = branch\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')
    script.run_script(self, '            $ brz config -d tree --remove file\n            $ brz config -d tree --all file\n            breezy:\n              [DEFAULT]\n              file = breezy\n            ')