import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def strip_multiple_whitespaces(s):
    """Remove repeating whitespace characters (spaces, tabs, line breaks) from `s`
    and turns tabs & line breaks into spaces using :const:`~gensim.parsing.preprocessing.RE_WHITESPACE`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string without repeating in a row whitespace characters.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_multiple_whitespaces
        >>> strip_multiple_whitespaces("salut" + '\\r' + " les" + '\\n' + "         loulous!")
        u'salut les loulous!'

    """
    s = utils.to_unicode(s)
    return RE_WHITESPACE.sub(' ', s)