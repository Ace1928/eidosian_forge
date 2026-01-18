from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
@classmethod
def create_standalone_workingtree(klass, base, format=None) -> 'WorkingTree':
    """Create a new ControlDir, WorkingTree, Branch and Repository at 'base'.

        'base' must be a local path or a file:// url.

        This will use the current default ControlDirFormat unless one is
        specified, and use whatever
        repository format that that uses for bzrdirformat.create_workingtree,
        create_branch and create_repository.

        Args:
          format: Override for the controldir format to create.

        Returns: The WorkingTree object.
        """
    t = _mod_transport.get_transport(base)
    from breezy.transport import local
    if not isinstance(t, local.LocalTransport):
        raise errors.NotLocalUrl(base)
    controldir = klass.create_branch_and_repo(base, force_new_repo=True, format=format).controldir
    return controldir.create_workingtree()