import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
@property
def push_bzr(self):
    """Return the push branch for this branch."""
    if self._push_bzr is None:
        self._push_bzr = branch.Branch.open(self.lp.bzr_identity)
    return self._push_bzr