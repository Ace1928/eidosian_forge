import re
from . import compat
from . import misc
def normalize_fragment(fragment):
    """Normalize the fragment string."""
    if not fragment:
        return fragment
    return normalize_percent_characters(fragment)