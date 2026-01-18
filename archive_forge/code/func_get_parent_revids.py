import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def get_parent_revids(self, revid):
    repo = self._branch.repository
    with repo.lock_read():
        retval = repo.get_parent_map([revid]).get(revid, None)
    return retval