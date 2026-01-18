import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds
def guess_encoding(data):
    """
    Given a byte string, attempt to decode it.
    Tries the standard 'UTF8' and 'latin-1' encodings,
    Plus several gathered from locale information.

    The calling program *must* first call::

        locale.setlocale(locale.LC_ALL, '')

    If successful it returns ``(decoded_unicode, successful_encoding)``.
    If unsuccessful it raises a ``UnicodeError``.
    """
    successful_encoding = None
    encodings = ['utf-8']
    try:
        encodings.append(locale.nl_langinfo(locale.CODESET))
    except AttributeError:
        pass
    try:
        encodings.append(locale.getlocale()[1])
    except (AttributeError, IndexError):
        pass
    try:
        encodings.append(locale.getdefaultlocale()[1])
    except (AttributeError, IndexError):
        pass
    encodings.append('latin-1')
    for enc in encodings:
        if not enc:
            continue
        try:
            decoded = str(data, enc)
            successful_encoding = enc
        except (UnicodeError, LookupError):
            pass
        else:
            break
    if not successful_encoding:
        raise UnicodeError('Unable to decode input data. Tried the following encodings: %s.' % ', '.join([repr(enc) for enc in encodings if enc]))
    else:
        return (decoded, successful_encoding)