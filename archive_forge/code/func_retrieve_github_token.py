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
def retrieve_github_token():
    auth_config = AuthenticationConfig()
    section = auth_config._get_config().get('Github')
    if section and section.get('private_token'):
        return section.get('private_token')
    path = os.path.join(bedding.config_dir(), 'github.conf')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip()