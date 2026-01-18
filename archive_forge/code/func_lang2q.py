import re
from warnings import warn
from nltk.corpus import bcp47
def lang2q(name):
    """
    Convert simple language name to Wikidata Q-code

    >>> lang2q('Low German')
    'Q25433'
    """
    return tag2q(langcode(name))