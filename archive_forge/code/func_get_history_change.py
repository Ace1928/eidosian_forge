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
def get_history_change(old_revision_id, new_revision_id, repository):
    """Calculate the uncommon lefthand history between two revisions.

    :param old_revision_id: The original revision id.
    :param new_revision_id: The new revision id.
    :param repository: The repository to use for the calculation.

    return old_history, new_history
    """
    old_history = []
    old_revisions = set()
    new_history = []
    new_revisions = set()
    graph = repository.get_graph()
    new_iter = graph.iter_lefthand_ancestry(new_revision_id)
    old_iter = graph.iter_lefthand_ancestry(old_revision_id)
    stop_revision = None
    do_old = True
    do_new = True
    while do_new or do_old:
        if do_new:
            try:
                new_revision = next(new_iter)
            except StopIteration:
                do_new = False
            else:
                new_history.append(new_revision)
                new_revisions.add(new_revision)
                if new_revision in old_revisions:
                    stop_revision = new_revision
                    break
        if do_old:
            try:
                old_revision = next(old_iter)
            except StopIteration:
                do_old = False
            else:
                old_history.append(old_revision)
                old_revisions.add(old_revision)
                if old_revision in new_revisions:
                    stop_revision = old_revision
                    break
    new_history.reverse()
    old_history.reverse()
    if stop_revision is not None:
        new_history = new_history[new_history.index(stop_revision) + 1:]
        old_history = old_history[old_history.index(stop_revision) + 1:]
    return (old_history, new_history)