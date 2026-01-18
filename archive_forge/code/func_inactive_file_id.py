import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def inactive_file_id(self, trans_id):
    """Return the inactive file_id associated with a transaction id.
        That is, the one in the tree or in non_present_ids.
        The file_id may actually be active, too.
        """
    file_id = self.tree_file_id(trans_id)
    if file_id is not None:
        return file_id
    for key, value in self._non_present_ids.items():
        if value == trans_id:
            return key