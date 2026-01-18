from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def create_clone_on_transport(self, to_transport, *, revision_id=None, stacked_on=None, create_prefix=False, use_existing_dir=False, no_tree=None, tag_selector=None):
    """Create a clone of this branch and its bzrdir.

        Args:
          to_transport: The transport to clone onto.
          revision_id: The revision id to use as tip in the new branch.
            If None the tip is obtained from this branch.
          stacked_on: An optional URL to stack the clone on.
          create_prefix: Create any missing directories leading up to
            to_transport.
          use_existing_dir: Use an existing directory if one exists.
        """
    if revision_id is None:
        revision_id = self.last_revision()
    dir_to = self.controldir.clone_on_transport(to_transport, revision_id=revision_id, stacked_on=stacked_on, create_prefix=create_prefix, use_existing_dir=use_existing_dir, no_tree=no_tree, tag_selector=tag_selector)
    return dir_to.open_branch()