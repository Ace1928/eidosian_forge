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
def get_proposal_by_url(self, url):
    try:
        owner, repo, pr_id = parse_github_pr_url(url)
    except NotGitHubUrl as e:
        raise UnsupportedForge(url) from e
    api_url = 'https://api.github.com/repos/{}/{}/pulls/{}'.format(owner, repo, pr_id)
    response = self._api_request('GET', api_url)
    if response.status != 200:
        raise UnexpectedHttpStatus(api_url, response.status, headers=response.getheaders())
    data = json.loads(response.text)
    return GitHubMergeProposal(self, data)