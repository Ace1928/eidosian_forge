import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
def iter_sibling_branches(control_dir, possible_transports=None):
    """Iterate over the siblings of a branch.

    :param control_dir: Control directory for which to look up the siblings
    :return: Iterator over tuples with branch name and branch object
    """
    try:
        reference = control_dir.get_branch_reference()
    except errors.NotBranchError:
        reference = None
    if reference is not None:
        try:
            ref_branch = Branch.open(reference, possible_transports=possible_transports)
        except errors.NotBranchError:
            ref_branch = None
    else:
        ref_branch = None
    if ref_branch is None or ref_branch.name:
        if ref_branch is not None:
            control_dir = ref_branch.controldir
        for name, branch in control_dir.get_branches().items():
            yield (name, branch)
    else:
        repo = ref_branch.controldir.find_repository()
        for branch in repo.find_branches(using=True):
            name = urlutils.relative_url(repo.user_url, branch.user_url).rstrip('/')
            yield (name, branch)