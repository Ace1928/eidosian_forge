import itertools
from contextlib import ExitStack
def generate_unicode_names():
    """Generate a sequence of arbitrary unique unicode names.

    By default they are not representable in ascii.

    >>> gen = generate_unicode_names()
    >>> n1 = next(gen)
    >>> n2 = next(gen)
    >>> type(n1)
    <class 'str'>
    >>> n1 == n2
    False
    >>> n1.encode('ascii', 'replace') == n1
    False
    """
    return ('∿%d' % x for x in itertools.count())