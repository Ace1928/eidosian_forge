import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def strip_non_alphanum(s):
    """Remove non-alphabetic characters from `s` using :const:`~gensim.parsing.preprocessing.RE_NONALPHA`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with alphabetic characters only.

    Notes
    -----
    Word characters - alphanumeric & underscore.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import strip_non_alphanum
        >>> strip_non_alphanum("if-you#can%read$this&then@this#method^works")
        u'if you can read this then this method works'

    """
    s = utils.to_unicode(s)
    return RE_NONALPHA.sub(' ', s)