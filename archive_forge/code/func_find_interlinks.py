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
def find_interlinks(raw):
    """Find all interlinks to other articles in the dump.

    Parameters
    ----------
    raw : str
        Unicode or utf-8 encoded string.

    Returns
    -------
    list
        List of tuples in format [(linked article, the actual text found), ...].

    """
    filtered = filter_wiki(raw, promote_remaining=False, simplify_links=False)
    interlinks_raw = re.findall(RE_P16, filtered)
    interlinks = []
    for parts in [i.split('|') for i in interlinks_raw]:
        actual_title = parts[0]
        try:
            interlink_text = parts[1]
        except IndexError:
            interlink_text = actual_title
        interlink_tuple = (actual_title, interlink_text)
        interlinks.append(interlink_tuple)
    legit_interlinks = [(i, j) for i, j in interlinks if '[' not in i and ']' not in i]
    return legit_interlinks