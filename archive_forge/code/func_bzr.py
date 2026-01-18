import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
@property
def bzr(self):
    """Return the bzr branch for this branch."""
    if self._bzr is None:
        self._bzr = branch.Branch.open(self.bzr_url)
    return self._bzr