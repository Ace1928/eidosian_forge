import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@property
def base_url(self):
    """
        Returns the base URL, given when the page was parsed.

        Use with ``urlparse.urljoin(el.base_url, href)`` to get
        absolute URLs.
        """
    return self.getroottree().docinfo.URL