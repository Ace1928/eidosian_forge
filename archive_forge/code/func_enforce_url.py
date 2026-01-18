from typing import (
from urllib.parse import urlparse
def enforce_url(value: Union['URL', bytes, str], *, name: str) -> 'URL':
    """
    Type check for URL parameters.
    """
    if isinstance(value, (bytes, str)):
        return URL(value)
    elif isinstance(value, URL):
        return value
    seen_type = type(value).__name__
    raise TypeError(f'{name} must be a URL, bytes, or str, but got {seen_type}.')