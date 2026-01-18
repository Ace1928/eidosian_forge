import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def is_merge_point(self, revid):
    return len(self.get_parent_revids(revid)) > 1