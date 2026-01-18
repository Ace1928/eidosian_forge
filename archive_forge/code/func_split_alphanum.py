import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def split_alphanum(s):
    """Add spaces between digits & letters in `s` using :const:`~gensim.parsing.preprocessing.RE_AL_NUM`.

    Parameters
    ----------
    s : str

    Returns
    -------
    str
        Unicode string with spaces between digits & letters.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import split_alphanum
        >>> split_alphanum("24.0hours7 days365 a1b2c3")
        u'24.0 hours 7 days 365 a 1 b 2 c 3'

    """
    s = utils.to_unicode(s)
    s = RE_AL_NUM.sub('\\1 \\2', s)
    return RE_NUM_AL.sub('\\1 \\2', s)