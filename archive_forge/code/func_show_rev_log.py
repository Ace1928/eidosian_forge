import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def show_rev_log(self, outf):
    """Write the current revision's log entry to a file."""
    rev = self._branch.repository.get_revision(self._revid)
    revno = '.'.join([str(x) for x in self.get_current_revno()])
    outf.write('On revision {} ({}):\n{}\n'.format(revno, rev.revision_id, rev.message))