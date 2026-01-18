import errno
import os
import shutil
from . import controldir, errors, ui
from .i18n import gettext
from .osutils import isdir
from .trace import note
from .workingtree import WorkingTree
def onerror(function, path, excinfo):
    """Show warning for errors seen by rmtree.
        """
    if function is not os.remove or excinfo[1].errno != errno.EACCES:
        raise
    ui.ui_factory.show_warning(gettext('unable to remove %s') % path)