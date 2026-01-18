import re
import unicodedata
def singularize(word):
    """
    Return the singular form of a word, the reverse of :func:`pluralize`.

    Examples::

        >>> singularize("posts")
        'post'
        >>> singularize("octopi")
        'octopus'
        >>> singularize("sheep")
        'sheep'
        >>> singularize("word")
        'word'
        >>> singularize("CamelOctopi")
        'CamelOctopus'

    """
    for inflection in UNCOUNTABLES:
        if re.search('(?i)\\b(%s)\\Z' % inflection, word):
            return word
    for rule, replacement in SINGULARS:
        if re.search(rule, word):
            return re.sub(rule, replacement, word)
    return word