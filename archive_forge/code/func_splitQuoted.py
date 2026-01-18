def splitQuoted(s):
    """
    Like a string split, but don't break substrings inside quotes.

    >>> splitQuoted('the "hairy monkey" likes pie')
    ['the', 'hairy monkey', 'likes', 'pie']

    Another one of those "someone must have a better solution for
    this" things.  This implementation is a VERY DUMB hack done too
    quickly.
    """
    out = []
    quot = None
    phrase = None
    for word in s.split():
        if phrase is None:
            if word and word[0] in ('"', "'"):
                quot = word[0]
                word = word[1:]
                phrase = []
        if phrase is None:
            out.append(word)
        elif word and word[-1] == quot:
            word = word[:-1]
            phrase.append(word)
            out.append(' '.join(phrase))
            phrase = None
        else:
            phrase.append(word)
    return out