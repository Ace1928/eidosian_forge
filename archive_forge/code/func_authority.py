from __future__ import annotations
import re
import typing
from ..exceptions import LocationParseError
from .util import to_str
@property
def authority(self) -> str | None:
    """
        Authority component as defined in RFC 3986 3.2.
        This includes userinfo (auth), host and port.

        i.e.
            userinfo@host:port
        """
    userinfo = self.auth
    netloc = self.netloc
    if netloc is None or userinfo is None:
        return netloc
    else:
        return f'{userinfo}@{netloc}'