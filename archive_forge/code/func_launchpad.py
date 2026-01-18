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
@property
def launchpad(self):
    if self._launchpad is None:
        self._launchpad = lp_api.connect_launchpad(self._api_base_url, version='devel')
    return self._launchpad