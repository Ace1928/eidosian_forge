import codecs
import sys
from future import utils
def register_surrogateescape():
    """
    Registers the surrogateescape error handler on Python 2 (only)
    """
    if utils.PY3:
        return
    try:
        codecs.lookup_error(FS_ERRORS)
    except LookupError:
        codecs.register_error(FS_ERRORS, surrogateescape_handler)