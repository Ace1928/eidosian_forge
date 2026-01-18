import re
from warnings import warn
from nltk.corpus import bcp47
def q2tag(qcode):
    """
    Convert Wikidata Q-code to BCP-47 tag

    >>> q2tag('Q4289225')
    'nds-u-sd-demv'
    """
    return wiki_bcp47[qcode]