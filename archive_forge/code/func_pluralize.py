import re
import unicodedata
def pluralize(word):
    """
    Return the plural form of a word.

    Examples::

        >>> pluralize("posts")
        'posts'
        >>> pluralize("octopus")
        'octopi'
        >>> pluralize("sheep")
        'sheep'
        >>> pluralize("CamelOctopus")
        'CamelOctopi'

    """
    if not word or word.lower() in UNCOUNTABLES:
        return word
    else:
        for rule, replacement in PLURALS:
            if re.search(rule, word):
                return re.sub(rule, replacement, word)
        return word