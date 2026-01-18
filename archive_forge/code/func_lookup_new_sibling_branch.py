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
def lookup_new_sibling_branch(control_dir, location, possible_transports=None):
    """Lookup the location for a new sibling branch.

    :param control_dir: Control directory to find sibling branches from
    :param location: Name of the new branch
    :return: Full location to the new branch
    """
    location = directory_service.directories.dereference(location)
    if '/' not in location and '\\' not in location:
        colocated, this_url = _is_colocated(control_dir, possible_transports)
        if colocated:
            return urlutils.join_segment_parameters(this_url, {'branch': urlutils.escape(location)})
        else:
            return urlutils.join(this_url, '..', urlutils.escape(location))
    return location