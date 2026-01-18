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
def cook_path_conflict(tt, fp, conflict_type, trans_id, file_id, this_parent, this_name, other_parent, other_name):
    if this_parent is None or this_name is None:
        this_path = '<deleted>'
    else:
        parent_path = fp.get_path(tt.trans_id_file_id(this_parent))
        this_path = osutils.pathjoin(parent_path, this_name)
    if other_parent is None or other_name is None:
        other_path = '<deleted>'
    else:
        try:
            parent_path = fp.get_path(tt.trans_id_file_id(other_parent))
        except NoFinalPath:
            parent_path = ''
        other_path = osutils.pathjoin(parent_path, other_name)
    return Conflict.factory(conflict_type, path=this_path, conflict_path=other_path, file_id=file_id)