import re
from warnings import warn
from nltk.corpus import bcp47
def tag2q(tag):
    """
    Convert BCP-47 tag to Wikidata Q-code

    >>> tag2q('nds-u-sd-demv')
    'Q4289225'
    """
    return bcp47.wiki_q[tag]