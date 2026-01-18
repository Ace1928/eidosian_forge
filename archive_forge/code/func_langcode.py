import re
from warnings import warn
from nltk.corpus import bcp47
def langcode(name, typ=2):
    """
    Convert language name to iso639-3 language code. Returns the short 2-letter
    code by default, if one is available, and the 3-letter code otherwise:

    >>> from nltk.langnames import langcode
    >>> langcode('Modern Greek (1453-)')
    'el'

    Specify 'typ=3' to get the 3-letter code:

    >>> langcode('Modern Greek (1453-)', typ=3)
    'ell'
    """
    if name in bcp47.langcode:
        code = bcp47.langcode[name]
        if typ == 3 and code in iso639long:
            code = iso639long[code]
        return code
    elif name in iso639code_retired:
        return iso639code_retired[name]
    else:
        warn(f'Could not find language in {name!r}', stacklevel=2)