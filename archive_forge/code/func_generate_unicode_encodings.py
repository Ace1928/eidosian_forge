import itertools
from contextlib import ExitStack
def generate_unicode_encodings(universal_encoding=None):
    """Return a generator of unicode encoding names.

    These can be passed to Python encode/decode/etc.

    :param universal_encoding: True/False/None tristate to say whether the
        generated encodings either can or cannot encode all unicode
        strings.

    >>> n1 = next(generate_unicode_names())
    >>> enc = next(generate_unicode_encodings(universal_encoding=True))
    >>> enc2 = next(generate_unicode_encodings(universal_encoding=False))
    >>> n1.encode(enc).decode(enc) == n1
    True
    >>> try:
    ...   n1.encode(enc2).decode(enc2)
    ... except UnicodeError:
    ...   print('fail')
    fail
    """
    if universal_encoding is not None:
        e = [n for n, u in interesting_encodings if u == universal_encoding]
    else:
        e = [n for n, u in interesting_encodings]
    return itertools.cycle(iter(e))