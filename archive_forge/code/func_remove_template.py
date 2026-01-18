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
def remove_template(s):
    """Remove template wikimedia markup.

    Parameters
    ----------
    s : str
        String containing markup template.

    Returns
    -------
    str
        Сopy of `s` with all the `wikimedia markup template <http://meta.wikimedia.org/wiki/Help:Template>`_ removed.

    Notes
    -----
    Since template can be nested, it is difficult remove them using regular expressions.

    """
    n_open, n_close = (0, 0)
    starts, ends = ([], [-1])
    in_template = False
    prev_c = None
    for i, c in enumerate(s):
        if not in_template:
            if c == '{' and c == prev_c:
                starts.append(i - 1)
                in_template = True
                n_open = 1
        if in_template:
            if c == '{':
                n_open += 1
            elif c == '}':
                n_close += 1
            if n_open == n_close:
                ends.append(i)
                in_template = False
                n_open, n_close = (0, 0)
        prev_c = c
    starts.append(None)
    return ''.join((s[end + 1:start] for end, start in zip(ends, starts)))