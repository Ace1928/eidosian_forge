import difflib
import patiencediff
from merge3 import Merge3
from ... import debug, merge, osutils
from ...trace import mutter
def file_matches(self, params):
    affected_files = self.affected_files
    if affected_files is None:
        config = self.merger.this_branch.get_config()
        config_key = self.name_prefix + '_merge_files'
        affected_files = config.get_user_option_as_list(config_key)
        if affected_files is None:
            affected_files = self.default_files
        self.affected_files = affected_files
    if affected_files:
        filepath = osutils.basename(params.this_path)
        if filepath in affected_files:
            return True
    return False