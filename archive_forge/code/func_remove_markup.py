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
def remove_markup(text, promote_remaining=True, simplify_links=True):
    """Filter out wiki markup from `text`, leaving only text.

    Parameters
    ----------
    text : str
        String containing markup.
    promote_remaining : bool
        Whether uncaught markup should be promoted to plain text.
    simplify_links : bool
        Whether links should be simplified keeping only their description text.

    Returns
    -------
    str
        `text` without markup.

    """
    text = re.sub(RE_P2, '', text)
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        old, iters = (text, iters + 1)
        text = re.sub(RE_P0, '', text)
        text = re.sub(RE_P1, '', text)
        text = re.sub(RE_P9, '', text)
        text = re.sub(RE_P10, '', text)
        text = re.sub(RE_P11, '', text)
        text = re.sub(RE_P14, '', text)
        text = re.sub(RE_P5, '\\3', text)
        if simplify_links:
            text = re.sub(RE_P6, '\\2', text)
        text = text.replace('!!', '\n|')
        text = text.replace('|-||', '\n|')
        text = re.sub(RE_P12, '\n', text)
        text = text.replace('|||', '|\n|')
        text = text.replace('||', '\n|')
        text = re.sub(RE_P13, '\n', text)
        text = re.sub(RE_P17, '\n', text)
        text = text.replace('[]', '')
        if old == text or iters > 2:
            break
    if promote_remaining:
        text = text.replace('[', '').replace(']', '')
    return text