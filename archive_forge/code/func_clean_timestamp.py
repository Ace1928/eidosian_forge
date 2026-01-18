import copy
import datetime
import keyword
import re
import unicodedata
import warnings
def clean_timestamp(dt=None, microseconds=False):
    """
    Return a timestamp that has been cleansed of characters that might
    cause problems in filenames, namely colons.  If no datetime object
    is provided, then uses the current time.

    The timestamp is in ISO-8601 format with the following exceptions:

    * Colons ':' are replaced by underscores '_'.
    * Microseconds are not displayed if the 'microseconds' parameter is
        False.

    .. deprecated:: 6.3.0
        This function will be removed in a future version of Traits.

    Parameters
    ----------
    dt : None or datetime.datetime
        If None, then the current time is used.
    microseconds : bool
        Display microseconds or not.

    Returns
    -------
    A string timestamp.
    """
    warnings.warn('clean_timestamp is deprecated and will eventually be removed', DeprecationWarning, stacklevel=2)
    if dt is None:
        dt = datetime.datetime.now()
    else:
        dt = copy.copy(dt)
    if not microseconds:
        dt = dt.replace(microsecond=0)
    stamp = dt.isoformat().replace(':', '_')
    return stamp