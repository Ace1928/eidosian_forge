import re
import string
import glob
from gensim import utils
from gensim.parsing.porter import PorterStemmer
def preprocess_documents(docs):
    """Apply :const:`~gensim.parsing.preprocessing.DEFAULT_FILTERS` to the documents strings.

    Parameters
    ----------
    docs : list of str

    Returns
    -------
    list of list of str
        Processed documents split by whitespace.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.parsing.preprocessing import preprocess_documents
        >>> preprocess_documents(["<i>Hel 9lo</i> <b>Wo9 rld</b>!", "Th3     weather_is really g00d today, isn't it?"])
        [[u'hel', u'rld'], [u'weather', u'todai', u'isn']]

    """
    return [preprocess_string(d) for d in docs]