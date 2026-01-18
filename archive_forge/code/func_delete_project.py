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
def delete_project(self, path):
    path = 'repos/' + path
    response = self._api_request('DELETE', path)
    if response.status == 404:
        raise NoSuchProject(path)
    if response.status == 204:
        return
    if response.status == 200:
        return json.loads(response.text)
    raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())