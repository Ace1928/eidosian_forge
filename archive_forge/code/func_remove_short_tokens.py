import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def remove_short_tokens(tokens, minsize=3):
    """Remove tokens shorter than `minsize` chars.

    Parameters
    ----------
    tokens : iterable of str
        Sequence of tokens.
    minsize : int, optimal
        Minimal length of token (include).

    Returns
    -------
    list of str
        List of tokens without short tokens.
    """
    return [token for token in tokens if len(token) >= minsize]