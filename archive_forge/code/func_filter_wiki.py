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
def filter_wiki(raw, promote_remaining=True, simplify_links=True):
    """Filter out wiki markup from `raw`, leaving only text.

    Parameters
    ----------
    raw : str
        Unicode or utf-8 encoded string.
    promote_remaining : bool
        Whether uncaught markup should be promoted to plain text.
    simplify_links : bool
        Whether links should be simplified keeping only their description text.

    Returns
    -------
    str
        `raw` without markup.

    """
    text = utils.to_unicode(raw, 'utf8', errors='ignore')
    text = utils.decode_htmlentities(text)
    return remove_markup(text, promote_remaining, simplify_links)