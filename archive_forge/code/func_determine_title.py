import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
def determine_title(description):
    """Determine the title for a merge proposal based on full description."""
    for firstline in description.splitlines():
        if firstline.strip():
            break
    else:
        raise ValueError
    try:
        i = firstline.index('. ')
    except ValueError:
        return firstline.rstrip('.')
    else:
        return firstline[:i]