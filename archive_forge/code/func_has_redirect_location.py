from __future__ import annotations
import datetime
import email.message
import json as jsonlib
import typing
import urllib.request
from collections.abc import Mapping
from http.cookiejar import Cookie, CookieJar
from ._content import ByteStream, UnattachedStream, encode_request, encode_response
from ._decoders import (
from ._exceptions import (
from ._multipart import get_multipart_boundary_from_content_type
from ._status_codes import codes
from ._types import (
from ._urls import URL
from ._utils import (
@property
def has_redirect_location(self) -> bool:
    """
        Returns True for 3xx responses with a properly formed URL redirection,
        `False` otherwise.
        """
    return self.status_code in (codes.MOVED_PERMANENTLY, codes.FOUND, codes.SEE_OTHER, codes.TEMPORARY_REDIRECT, codes.PERMANENT_REDIRECT) and 'Location' in self.headers