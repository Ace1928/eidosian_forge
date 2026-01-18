import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def retire_bzrdir(self, limit=10000):
    """Permanently disable the bzrdir.

        This is done by renaming it to give the user some ability to recover
        if there was a problem.

        This will have horrible consequences if anyone has anything locked or
        in use.
        :param limit: number of times to retry
        """
    i = 0
    while True:
        try:
            to_path = '.bzr.retired.%d' % i
            self.root_transport.rename('.bzr', to_path)
            note(gettext('renamed {0} to {1}').format(self.root_transport.abspath('.bzr'), to_path))
            return
        except (errors.TransportError, OSError, errors.PathError):
            i += 1
            if i > limit:
                raise
            else:
                pass