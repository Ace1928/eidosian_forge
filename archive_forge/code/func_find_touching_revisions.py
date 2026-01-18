import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def find_touching_revisions(repository, last_revision, last_tree, last_path):
    """Yield a description of revisions which affect the file.

    Each returned element is (revno, revision_id, description)

    This is the list of revisions where the file is either added,
    modified, renamed or deleted.

    TODO: Perhaps some way to limit this to only particular revisions,
    or to traverse a non-mainline set of revisions?
    """
    last_verifier = last_tree.get_file_verifier(last_path)
    graph = repository.get_graph()
    history = list(graph.iter_lefthand_ancestry(last_revision, []))
    revno = len(history)
    for revision_id in history:
        this_tree = repository.revision_tree(revision_id)
        this_intertree = InterTree.get(this_tree, last_tree)
        this_path = this_intertree.find_source_path(last_path)
        if this_path is not None and last_path is None:
            yield (revno, revision_id, 'deleted ' + this_path)
            this_verifier = this_tree.get_file_verifier(this_path)
        elif this_path is None and last_path is not None:
            yield (revno, revision_id, 'added ' + last_path)
        elif this_path != last_path:
            yield (revno, revision_id, 'renamed {} => {}'.format(this_path, last_path))
            this_verifier = this_tree.get_file_verifier(this_path)
        else:
            this_verifier = this_tree.get_file_verifier(this_path)
            if this_verifier != last_verifier:
                yield (revno, revision_id, 'modified ' + this_path)
        last_verifier = this_verifier
        last_path = this_path
        last_tree = this_tree
        if last_path is None:
            return
        revno -= 1