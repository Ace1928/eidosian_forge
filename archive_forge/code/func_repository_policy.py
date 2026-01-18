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
def repository_policy(found_bzrdir):
    stack_on = None
    stack_on_pwd = None
    config = found_bzrdir.get_config()
    stop = False
    stack_on = config.get_default_stack_on()
    if stack_on is not None:
        stack_on_pwd = found_bzrdir.user_url
        stop = True
    try:
        repository = found_bzrdir.open_repository()
    except errors.NoRepositoryPresent:
        repository = None
    else:
        if found_bzrdir.user_url != self.user_url and (not repository.is_shared()):
            repository = None
            stop = True
        else:
            stop = True
    if not stop:
        return (None, False)
    if repository:
        return (UseExistingRepository(repository, stack_on, stack_on_pwd, require_stacking=require_stacking), True)
    else:
        return (CreateRepository(self, stack_on, stack_on_pwd, require_stacking=require_stacking), True)