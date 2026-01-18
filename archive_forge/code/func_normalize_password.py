import re
from . import compat
from . import misc
def normalize_password(password):
    """Normalize a password to make safe for userinfo."""
    return compat.urlquote(password)