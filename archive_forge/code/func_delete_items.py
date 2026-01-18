import errno
import os
import shutil
from . import controldir, errors, ui
from .i18n import gettext
from .osutils import isdir
from .trace import note
from .workingtree import WorkingTree
def delete_items(deletables, dry_run=False):
    """Delete files in the deletables iterable"""

    def onerror(function, path, excinfo):
        """Show warning for errors seen by rmtree.
        """
        if function is not os.remove or excinfo[1].errno != errno.EACCES:
            raise
        ui.ui_factory.show_warning(gettext('unable to remove %s') % path)
    has_deleted = False
    for path, subp in deletables:
        if not has_deleted:
            note(gettext('deleting paths:'))
            has_deleted = True
        if not dry_run:
            if isdir(path):
                shutil.rmtree(path, onerror=onerror)
            else:
                try:
                    os.unlink(path)
                    note('  ' + subp)
                except OSError as e:
                    if e.errno != errno.EACCES:
                        raise e
                    ui.ui_factory.show_warning(gettext('unable to remove "{0}": {1}.').format(path, e.strerror))
        else:
            note('  ' + subp)
    if not has_deleted:
        note(gettext('No files deleted.'))