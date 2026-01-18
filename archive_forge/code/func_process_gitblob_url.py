from __future__ import annotations
import re
from typing import Final, Literal
from urllib.parse import urlparse
from typing_extensions import TypeAlias
def process_gitblob_url(url: str) -> str:
    """Check url to see if it describes a GitHub Gist "blob" URL.

    If so, returns a new URL to get the "raw" script.
    If not, returns URL unchanged.
    """
    match = _GITBLOB_RE.match(url)
    if match:
        mdict = match.groupdict()
        if mdict['blob_or_raw'] == 'blob':
            return '{base}{account}raw{suffix}'.format(**mdict)
        if mdict['blob_or_raw'] == 'raw':
            return url
        return url + '/raw'
    return url