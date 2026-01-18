import contextlib
from . import errors
from .controldir import ControlDir
from .i18n import gettext
from .trace import note
def scan_branch(branch, needed_refs, exit_stack):
    """Scan a branch for refs.

    :param branch:  The branch to schedule for checking.
    :param needed_refs: Refs we are accumulating.
    :param exit_stack: The exit stack accumulating.
    """
    note(gettext("Checking branch at '%s'.") % (branch.base,))
    exit_stack.enter_context(branch.lock_read())
    branch_refs = branch._get_check_refs()
    for ref in branch_refs:
        reflist = needed_refs.setdefault(ref, [])
        reflist.append(branch)