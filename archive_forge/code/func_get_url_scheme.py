import os
import string
import urllib.parse
import urllib.request
from typing import Optional
from .compat import WINDOWS
def get_url_scheme(url: str) -> Optional[str]:
    if ':' not in url:
        return None
    return url.split(':', 1)[0].lower()