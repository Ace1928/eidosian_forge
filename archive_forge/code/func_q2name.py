import re
from warnings import warn
from nltk.corpus import bcp47
def q2name(qcode, typ='full'):
    """
    Convert Wikidata Q-code to BCP-47 (full or short) language name

    >>> q2name('Q4289225')
    'Low German: Mecklenburg-Vorpommern'

    >>> q2name('Q4289225', "short")
    'Low German'
    """
    return langname(q2tag(qcode), typ)