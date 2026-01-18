import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ... import version_string as breezy_version
from ...config import AuthenticationConfig, GlobalStack
from ...errors import (InvalidHttpResponse, PermissionDenied,
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...i18n import gettext
from ...trace import note
from ...transport import get_transport
from ...transport.http import default_user_agent
def iter_proposals(self, source_branch, target_branch, status='open'):
    source_owner, source_repo_name, source_branch_name = parse_github_branch_url(source_branch)
    target_owner, target_repo_name, target_branch_name = parse_github_branch_url(target_branch)
    target_repo = self._get_repo(target_owner, target_repo_name)
    state = {'open': 'open', 'merged': 'closed', 'closed': 'closed', 'all': 'all'}
    pulls = self._get_repo_pulls(strip_optional(target_repo['pulls_url']), head=target_branch_name, state=state[status])
    for pull in pulls:
        if status == 'closed' and pull['merged'] or (status == 'merged' and (not pull['merged'])):
            continue
        if pull['head']['ref'] != source_branch_name:
            continue
        if pull['head']['repo'] is None:
            continue
        if pull['head']['repo']['owner']['login'] != source_owner or pull['head']['repo']['name'] != source_repo_name:
            continue
        yield GitHubMergeProposal(self, pull)