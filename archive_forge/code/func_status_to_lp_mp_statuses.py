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
def status_to_lp_mp_statuses(status):
    statuses = []
    if status in ('open', 'all'):
        statuses.extend(['Work in progress', 'Needs review', 'Approved', 'Code failed to merge', 'Queued'])
    if status in ('closed', 'all'):
        statuses.extend(['Rejected', 'Superseded'])
    if status in ('merged', 'all'):
        statuses.append('Merged')
    return statuses