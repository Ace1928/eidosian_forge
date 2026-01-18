import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
def parse_launchpadlib_version(version_number):
    """Parse a version number of the style used by launchpadlib."""
    return tuple(map(int, version_number.split('.')))