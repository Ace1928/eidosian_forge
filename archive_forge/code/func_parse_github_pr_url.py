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
def parse_github_pr_url(url):
    scheme, user, password, host, port, path = urlutils.parse_url(url)
    if host != GITHUB_HOST:
        raise NotGitHubUrl(url)
    try:
        owner, repo_name, pull, pr_id = path.strip('/').split('/')
    except IndexError as e:
        raise ValueError('Not a PR URL') from e
    if pull != 'pull':
        raise ValueError('Not a PR URL')
    return (owner, repo_name, pr_id)