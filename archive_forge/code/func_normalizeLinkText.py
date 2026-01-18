from __future__ import annotations
from collections.abc import Callable
from contextlib import suppress
import re
from urllib.parse import quote, unquote, urlparse, urlunparse  # noqa: F401
import mdurl
from .. import _punycode
def normalizeLinkText(url: str) -> str:
    """Normalize autolink content

    ::

        <destination>
         ~~~~~~~~~~~
    """
    parsed = mdurl.parse(url, slashes_denote_host=True)
    if parsed.hostname and (not parsed.protocol or parsed.protocol in RECODE_HOSTNAME_FOR):
        with suppress(Exception):
            parsed = parsed._replace(hostname=_punycode.to_unicode(parsed.hostname))
    return mdurl.decode(mdurl.format(parsed), mdurl.DECODE_DEFAULT_CHARS + '%')