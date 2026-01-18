import sys
from typing import Iterator, Optional
from urllib.parse import ParseResult, urlparse
from .config import ConfigDict, SectionLike
def match_partial_url(valid_url: ParseResult, partial_url: str) -> bool:
    """Matches a parsed url with a partial url (no scheme/netloc)."""
    if '://' not in partial_url:
        parsed = urlparse('scheme://' + partial_url)
    else:
        parsed = urlparse(partial_url)
        if valid_url.scheme != parsed.scheme:
            return False
    if any((parsed.hostname and valid_url.hostname != parsed.hostname, parsed.username and valid_url.username != parsed.username, parsed.port and valid_url.port != parsed.port, parsed.path and parsed.path.rstrip('/') != valid_url.path.rstrip('/'))):
        return False
    return True