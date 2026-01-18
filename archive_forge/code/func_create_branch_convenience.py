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
def create_branch_convenience(klass, base, force_new_repo=False, force_new_tree=None, format=None, possible_transports=None):
    """Create a new ControlDir, Branch and Repository at the url 'base'.

        This is a convenience function - it will use an existing repository
        if possible, can be told explicitly whether to create a working tree or
        not.

        This will use the current default ControlDirFormat unless one is
        specified, and use whatever
        repository format that that uses via ControlDir.create_branch and
        create_repository. If a shared repository is available that is used
        preferentially. Whatever repository is used, its tree creation policy
        is followed.

        The created Branch object is returned.
        If a working tree cannot be made due to base not being a file:// url,
        no error is raised unless force_new_tree is True, in which case no
        data is created on disk and NotLocalUrl is raised.

        Args:
          base: The URL to create the branch at.
          force_new_repo: If True a new repository is always created.
          force_new_tree: If True or False force creation of a tree or
                               prevent such creation respectively.
          format: Override for the controldir format to create.
          possible_transports: An optional reusable transports list.
        """
    if force_new_tree:
        from breezy.transport import local
        t = _mod_transport.get_transport(base, possible_transports)
        if not isinstance(t, local.LocalTransport):
            raise errors.NotLocalUrl(base)
    controldir = klass.create(base, format, possible_transports)
    repo = controldir._find_or_create_repository(force_new_repo)
    result = controldir.create_branch()
    if force_new_tree or (repo.make_working_trees() and force_new_tree is None):
        try:
            controldir.create_workingtree()
        except errors.NotLocalUrl:
            pass
    return result