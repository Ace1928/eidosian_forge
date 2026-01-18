import re
from . import compat
from . import misc
def normalize_query(query):
    """Normalize the query string."""
    if not query:
        return query
    return normalize_percent_characters(query)