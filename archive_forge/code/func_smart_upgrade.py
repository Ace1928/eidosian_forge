from . import errors, trace, ui, urlutils
from .bzr.remote import RemoteBzrDir
from .controldir import ControlDir, format_registry
from .i18n import gettext
def smart_upgrade(control_dirs, format, clean_up=False, dry_run=False):
    """Convert control directories to a new format intelligently.

    If the control directory is a shared repository, dependent branches
    are also converted provided the repository converted successfully.
    If the conversion of a branch fails, remaining branches are still tried.

    :param control_dirs: the BzrDirs to upgrade
    :param format: the format to convert to or None for the best default
    :param clean_up: if True, the backup.bzr directory is removed if the
      upgrade succeeded for a given repo/branch/tree
    :param dry_run: show what would happen but don't actually do any upgrades
    :return: attempted-control-dirs, succeeded-control-dirs, exceptions
    """
    all_attempted = []
    all_succeeded = []
    all_exceptions = []
    for control_dir in control_dirs:
        attempted, succeeded, exceptions = _smart_upgrade_one(control_dir, format, clean_up=clean_up, dry_run=dry_run)
        all_attempted.extend(attempted)
        all_succeeded.extend(succeeded)
        all_exceptions.extend(exceptions)
    return (all_attempted, all_succeeded, all_exceptions)