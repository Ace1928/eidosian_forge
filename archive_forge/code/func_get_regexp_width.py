import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def get_regexp_width(expr: str) -> Union[Tuple[int, int], List[int]]:
    if _has_regex:
        regexp_final = re.sub(categ_pattern, 'A', expr)
    else:
        if re.search(categ_pattern, expr):
            raise ImportError('`regex` module must be installed in order to use Unicode categories.', expr)
        regexp_final = expr
    try:
        return [int(x) for x in sre_parse.parse(regexp_final).getwidth()]
    except sre_constants.error:
        if not _has_regex:
            raise ValueError(expr)
        else:
            c = regex.compile(regexp_final)
            MAXWIDTH = getattr(sre_parse, 'MAXWIDTH', sre_constants.MAXREPEAT)
            if c.match('') is None:
                return (1, int(MAXWIDTH))
            else:
                return (0, int(MAXWIDTH))