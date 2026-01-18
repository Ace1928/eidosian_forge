from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_all_remove_exclusive(self):
    self.run_bzr_error(['--all and --remove are mutually exclusive.'], ['config', '--remove', '--all'])