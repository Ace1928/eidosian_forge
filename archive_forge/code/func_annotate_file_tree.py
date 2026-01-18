import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def annotate_file_tree(tree, path, to_file, verbose=False, full=False, show_ids=False, branch=None):
    """Annotate path in a tree.

    The tree should already be read_locked() when annotate_file_tree is called.

    :param tree: The tree to look for revision numbers and history from.
    :param path: The path to annotate
    :param to_file: The file to output the annotation to.
    :param verbose: Show all details rather than truncating to ensure
        reasonable text width.
    :param full: XXXX Not sure what this does.
    :param show_ids: Show revision ids in the annotation output.
    :param branch: Branch to use for revision revno lookups
    """
    if branch is None:
        branch = tree.branch
    if to_file is None:
        to_file = sys.stdout
    encoding = osutils.get_terminal_encoding()
    annotations = list(tree.annotate_iter(path))
    if show_ids:
        return _show_id_annotations(annotations, to_file, full, encoding)
    if not getattr(tree, 'get_revision_id', False):
        current_rev = Revision(CURRENT_REVISION)
        current_rev.parent_ids = tree.get_parent_ids()
        try:
            current_rev.committer = branch.get_config_stack().get('email')
        except errors.NoWhoami:
            current_rev.committer = 'local user'
        current_rev.message = '?'
        current_rev.timestamp = round(time.time(), 3)
        current_rev.timezone = osutils.local_time_offset()
    else:
        current_rev = None
    annotation = list(_expand_annotations(annotations, branch, current_rev))
    _print_annotations(annotation, verbose, to_file, full, encoding)