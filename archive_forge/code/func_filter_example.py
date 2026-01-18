import bz2
import logging
import multiprocessing
import re
import signal
from pickle import PicklingError
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
def filter_example(elem, text, *args, **kwargs):
    """Example function for filtering arbitrary documents from wikipedia dump.


    The custom filter function is called _before_ tokenisation and should work on
    the raw text and/or XML element information.

    The filter function gets the entire context of the XML element passed into it,
    but you can of course choose not the use some or all parts of the context. Please
    refer to :func:`gensim.corpora.wikicorpus.extract_pages` for the exact details
    of the page context.

    Parameters
    ----------
    elem : etree.Element
        XML etree element
    text : str
        The text of the XML node
    namespace : str
        XML namespace of the XML element
    title : str
       Page title
    page_tag : str
        XPath expression for page.
    text_path : str
        XPath expression for text.
    title_path : str
        XPath expression for title.
    ns_path : str
        XPath expression for namespace.
    pageid_path : str
        XPath expression for page id.

    Example
    -------
    .. sourcecode:: pycon

        >>> import gensim.corpora
        >>> filter_func = gensim.corpora.wikicorpus.filter_example
        >>> dewiki = gensim.corpora.WikiCorpus(
        ...     './dewiki-20180520-pages-articles-multistream.xml.bz2',
        ...     filter_articles=filter_func)

    """
    _regex_de_excellent = re.compile('.*\\{\\{(Exzellent.*?)\\}\\}[\\s]*', flags=re.DOTALL)
    _regex_de_featured = re.compile('.*\\{\\{(Lesenswert.*?)\\}\\}[\\s]*', flags=re.DOTALL)
    if text is None:
        return False
    if _regex_de_excellent.match(text) or _regex_de_featured.match(text):
        return True
    else:
        return False