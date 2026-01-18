import re
import sys
from ast import literal_eval
from functools import total_ordering
from typing import NamedTuple, Sequence, Union
def python_bytes_to_unicode(source: Union[str, bytes], encoding: str='utf-8', errors: str='strict') -> str:
    """
    Checks for unicode BOMs and PEP 263 encoding declarations. Then returns a
    unicode object like in :py:meth:`bytes.decode`.

    :param encoding: See :py:meth:`bytes.decode` documentation.
    :param errors: See :py:meth:`bytes.decode` documentation. ``errors`` can be
        ``'strict'``, ``'replace'`` or ``'ignore'``.
    """

    def detect_encoding():
        """
        For the implementation of encoding definitions in Python, look at:
        - http://www.python.org/dev/peps/pep-0263/
        - http://docs.python.org/2/reference/lexical_analysis.html#encoding-declarations
        """
        byte_mark = literal_eval("b'\\xef\\xbb\\xbf'")
        if source.startswith(byte_mark):
            return 'utf-8'
        first_two_lines = re.match(b'(?:[^\\r\\n]*(?:\\r\\n|\\r|\\n)){0,2}', source).group(0)
        possible_encoding = re.search(b'coding[=:]\\s*([-\\w.]+)', first_two_lines)
        if possible_encoding:
            e = possible_encoding.group(1)
            if not isinstance(e, str):
                e = str(e, 'ascii', 'replace')
            return e
        else:
            return encoding
    if isinstance(source, str):
        return source
    encoding = detect_encoding()
    try:
        return str(source, encoding, errors)
    except LookupError:
        if errors == 'replace':
            return str(source, 'utf-8', errors)
        raise