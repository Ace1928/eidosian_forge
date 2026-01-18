import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
def store_gitlab_token(name, url, private_token):
    """Store a GitLab token in a configuration file."""
    from breezy.config import AuthenticationConfig
    auth_config = AuthenticationConfig()
    auth_config._set_option(name, 'url', url)
    auth_config._set_option(name, 'private_token', private_token)