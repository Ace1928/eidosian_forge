import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
def get_trees_and_branches_to_diff_locked(path_list, revision_specs, old_url, new_url, exit_stack, apply_view=True):
    """Get the trees and specific files to diff given a list of paths.

    This method works out the trees to be diff'ed and the files of
    interest within those trees.

    :param path_list:
        the list of arguments passed to the diff command
    :param revision_specs:
        Zero, one or two RevisionSpecs from the diff command line,
        saying what revisions to compare.
    :param old_url:
        The url of the old branch or tree. If None, the tree to use is
        taken from the first path, if any, or the current working tree.
    :param new_url:
        The url of the new branch or tree. If None, the tree to use is
        taken from the first path, if any, or the current working tree.
    :param exit_stack:
        an ExitStack object. get_trees_and_branches_to_diff
        will register cleanups that must be run to unlock the trees, etc.
    :param apply_view:
        if True and a view is set, apply the view or check that the paths
        are within it
    :returns:
        a tuple of (old_tree, new_tree, old_branch, new_branch,
        specific_files, extra_trees) where extra_trees is a sequence of
        additional trees to search in for file-ids.  The trees and branches
        will be read-locked until the cleanups registered via the exit_stack
        param are run.
    """
    old_revision_spec = None
    new_revision_spec = None
    if revision_specs is not None:
        if len(revision_specs) > 0:
            old_revision_spec = revision_specs[0]
            if old_url is None:
                old_url = old_revision_spec.get_branch()
        if len(revision_specs) > 1:
            new_revision_spec = revision_specs[1]
            if new_url is None:
                new_url = new_revision_spec.get_branch()
    other_paths = []
    make_paths_wt_relative = True
    consider_relpath = True
    if path_list is None or len(path_list) == 0:
        default_location = '.'
        consider_relpath = False
    elif old_url is not None and new_url is not None:
        other_paths = path_list
        make_paths_wt_relative = False
    else:
        default_location = path_list[0]
        other_paths = path_list[1:]

    def lock_tree_or_branch(wt, br):
        if wt is not None:
            exit_stack.enter_context(wt.lock_read())
        elif br is not None:
            exit_stack.enter_context(br.lock_read())
    specific_files = []
    if old_url is None:
        old_url = default_location
    working_tree, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch(old_url)
    lock_tree_or_branch(working_tree, branch)
    if consider_relpath and relpath != '':
        if working_tree is not None and apply_view:
            views.check_path_in_view(working_tree, relpath)
        specific_files.append(relpath)
    old_tree = _get_tree_to_diff(old_revision_spec, working_tree, branch)
    old_branch = branch
    if new_url is None:
        new_url = default_location
    if new_url != old_url:
        working_tree, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch(new_url)
        lock_tree_or_branch(working_tree, branch)
        if consider_relpath and relpath != '':
            if working_tree is not None and apply_view:
                views.check_path_in_view(working_tree, relpath)
            specific_files.append(relpath)
    new_tree = _get_tree_to_diff(new_revision_spec, working_tree, branch, basis_is_default=working_tree is None)
    new_branch = branch
    if make_paths_wt_relative and working_tree is not None:
        other_paths = working_tree.safe_relpath_files(other_paths, apply_view=apply_view)
    specific_files.extend(other_paths)
    if len(specific_files) == 0:
        specific_files = None
        if working_tree is not None and working_tree.supports_views() and apply_view:
            view_files = working_tree.views.lookup_view()
            if view_files:
                specific_files = view_files
                view_str = views.view_display_str(view_files)
                note(gettext('*** Ignoring files outside view. View is %s') % view_str)
    extra_trees = None
    if working_tree is not None and working_tree not in (old_tree, new_tree):
        extra_trees = (working_tree,)
    return (old_tree, new_tree, old_branch, new_branch, specific_files, extra_trees)