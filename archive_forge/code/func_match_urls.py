import sys
from typing import Iterator, Optional
from urllib.parse import ParseResult, urlparse
from .config import ConfigDict, SectionLike
def match_urls(url: ParseResult, url_prefix: ParseResult) -> bool:
    base_match = url.scheme == url_prefix.scheme and url.hostname == url_prefix.hostname and (url.port == url_prefix.port)
    user_match = url.username == url_prefix.username if url_prefix.username else True
    path_match = url.path.rstrip('/').startswith(url_prefix.path.rstrip())
    return base_match and user_match and path_match