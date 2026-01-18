from __future__ import annotations
import codecs
import collections
import datetime
import os
import re
import textwrap
from collections.abc import Generator, Iterable
from typing import IO, Any, TypeVar
from babel import dates, localtime
def parse_future_flags(fp: IO[bytes], encoding: str='latin-1') -> int:
    """Parse the compiler flags by :mod:`__future__` from the given Python
    code.
    """
    import __future__
    pos = fp.tell()
    fp.seek(0)
    flags = 0
    try:
        body = fp.read().decode(encoding)
        body = re.sub('import\\s*\\([\\r\\n]+', 'import (', body)
        body = re.sub(',\\s*[\\r\\n]+', ', ', body)
        body = re.sub('\\\\\\s*[\\r\\n]+', ' ', body)
        for m in PYTHON_FUTURE_IMPORT_re.finditer(body):
            names = [x.strip().strip('()') for x in m.group(1).split(',')]
            for name in names:
                feature = getattr(__future__, name, None)
                if feature:
                    flags |= feature.compiler_flag
    finally:
        fp.seek(pos)
    return flags