import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def dirname(url: str, exclude_trailing_slash: bool=True) -> str:
    """Return the parent directory of the given path.

    Args:
      url: Relative or absolute URL
      exclude_trailing_slash: Remove a final slash
        (treat http://host/foo/ as http://host/foo, but
        http://host/ stays http://host/)
    Returns: Everything in the URL except the last path chunk
    """
    return split(url, exclude_trailing_slash=exclude_trailing_slash)[0]