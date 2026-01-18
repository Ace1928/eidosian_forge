from re import finditer
from xml.sax.saxutils import escape, unescape
def regexp_span_tokenize(s, regexp):
    """
    Return the offsets of the tokens in *s*, as a sequence of ``(start, end)``
    tuples, by splitting the string at each successive match of *regexp*.

        >>> from nltk.tokenize.util import regexp_span_tokenize
        >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me
        ... two of them.\\n\\nThanks.'''
        >>> list(regexp_span_tokenize(s, r'\\s')) # doctest: +NORMALIZE_WHITESPACE
        [(0, 4), (5, 12), (13, 17), (18, 23), (24, 26), (27, 30), (31, 36),
        (38, 44), (45, 48), (49, 51), (52, 55), (56, 58), (59, 64), (66, 73)]

    :param s: the string to be tokenized
    :type s: str
    :param regexp: regular expression that matches token separators (must not be empty)
    :type regexp: str
    :rtype: iter(tuple(int, int))
    """
    left = 0
    for m in finditer(regexp, s):
        right, next = m.span()
        if right != left:
            yield (left, right)
        left = next
    yield (left, len(s))