import re
from . import errors
def lazy_compile(*args, **kwargs):
    """Create a proxy object which will compile the regex on demand.

    :return: a LazyRegex proxy object.
    """
    return LazyRegex(args, kwargs)