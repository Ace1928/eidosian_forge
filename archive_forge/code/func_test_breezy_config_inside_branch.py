from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_breezy_config_inside_branch(self):
    script.run_script(self, '            $ brz config -d tree --scope breezy --remove file\n            $ brz config -d tree --all file\n            locations:\n              [.../work/tree]\n              file = locations\n            branch:\n              file = branch\n            ')