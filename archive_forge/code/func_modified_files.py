import re
import shutil
import tempfile
from typing import Any, List, Optional
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ...forge import (AutoMergeUnsupported, Forge, LabelsUnsupported,
from ...git.urls import git_url_to_bzr_url
from ...lazy_import import lazy_import
from ...trace import mutter
from breezy.plugins.launchpad import (
from ...transport import get_transport
def modified_files(old_tree, new_tree):
    """Return a list of paths in the new tree with modified contents."""
    for change in new_tree.iter_changes(old_tree):
        if change.changed_content and change.kind[1] == 'file':
            yield str(path)