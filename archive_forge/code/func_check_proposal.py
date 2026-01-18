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
def check_proposal(self):
    """Check that the submission is sensible."""
    if self.source_branch_lp.self_link == self.target_branch_lp.self_link:
        raise errors.CommandError('Source and target branches must be different.')
    for mp in self.source_branch_lp.landing_targets:
        if mp.queue_status in ('Merged', 'Rejected'):
            continue
        if mp.target_branch.self_link == self.target_branch_lp.self_link:
            raise MergeProposalExists(lp_uris.canonical_url(mp))