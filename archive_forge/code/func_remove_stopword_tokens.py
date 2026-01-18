import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def remove_stopword_tokens(tokens, stopwords=None):
    """Remove stopword tokens using list `stopwords`.

    Parameters
    ----------
    tokens : iterable of str
        Sequence of tokens.
    stopwords : iterable of str, optional
        Sequence of stopwords
        If None - using :const:`~gensim.parsing.preprocessing.STOPWORDS`

    Returns
    -------
    list of str
        List of tokens without `stopwords`.

    """
    if stopwords is None:
        stopwords = STOPWORDS
    return [token for token in tokens if token not in stopwords]