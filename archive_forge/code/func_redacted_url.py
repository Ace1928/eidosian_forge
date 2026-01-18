import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
@property
def redacted_url(self) -> str:
    """url with user:password part removed unless it is formed with
        environment variables as specified in PEP 610, or it is ``git``
        in the case of a git URL.
        """
    purl = urllib.parse.urlsplit(self.url)
    netloc = self._remove_auth_from_netloc(purl.netloc)
    surl = urllib.parse.urlunsplit((purl.scheme, netloc, purl.path, purl.query, purl.fragment))
    return surl