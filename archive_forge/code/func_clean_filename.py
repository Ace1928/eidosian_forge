import copy
import datetime
import keyword
import re
import unicodedata
import warnings
def clean_filename(name, replace_empty=''):
    """
    Make a user-supplied string safe for filename use.

    Returns an ASCII-encodable string based on the input string that's safe for
    use as a component of a filename or URL. The returned value is a string
    containing only lowercase ASCII letters, digits, and the characters '-' and
    '_'.

    This does not give a faithful representation of the original string:
    different input strings can result in the same output string.

    .. deprecated:: 6.3.0
        This function will be removed in a future version of Traits.

    Parameters
    ----------
    name : str
        The string to be made safe.
    replace_empty : str, optional
        The return value to be used in the event that the sanitised
        string ends up being empty. No validation is done on this
        input - it's up to the user to ensure that the default is
        itself safe. The default is to return the empty string.

    Returns
    -------
    safe_string : str
        A filename-safe version of string.

    """
    warnings.warn('clean_filename is deprecated and will eventually be removed', DeprecationWarning, stacklevel=2)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub('[^\\w\\s-]', '', name).strip().lower()
    safe_name = re.sub('[-\\s]+', '-', name)
    if safe_name == '':
        return replace_empty
    return safe_name