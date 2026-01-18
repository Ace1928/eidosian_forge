import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def remove_stopwords(s, stopwords=None):
    """Remove :const:`~gensim.parsing.preprocessing.STOPWORDS` from `s`.

    Parameters
    ----------
    s : str
    stopwords : iterable of str, optional
        Sequence of stopwords
        If None - using :const:`~gensim.parsing.preprocessing.STOPWORDS`

    Returns
    -------
    str
        Unicode string without `stopwords`.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import remove_stopwords
        >>> remove_stopwords("Better late than never, but better never late.")
        u'Better late never, better late.'

    """
    s = utils.to_unicode(s)
    return ' '.join(remove_stopword_tokens(s.split(), stopwords))