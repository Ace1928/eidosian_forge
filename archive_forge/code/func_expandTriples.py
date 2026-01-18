from __future__ import annotations
import re
import sys
from typing import Any, BinaryIO, List
from typing import Optional as OptionalType
from typing import TextIO, Tuple, Union
from pyparsing import CaselessKeyword as Keyword  # watch out :)
from pyparsing import (
import rdflib
from rdflib.compat import decodeUnicodeEscape
from . import operators as op
from .parserutils import Comp, CompValue, Param, ParamList
def expandTriples(terms: ParseResults) -> List[Any]:
    """
    Expand ; and , syntax for repeat predicates, subjects
    """
    try:
        res: List[Any] = []
        if DEBUG:
            print('Terms', terms)
        l_ = len(terms)
        for i, t in enumerate(terms):
            if t == ',':
                res.extend([res[-3], res[-2]])
            elif t == ';':
                if i + 1 == len(terms) or terms[i + 1] == ';' or terms[i + 1] == '.':
                    continue
                res.append(res[0])
            elif isinstance(t, list):
                if len(res) % 3 == 2:
                    res.append(t[0])
                if len(t) > 1:
                    res += t
                if i + 1 < l_ and terms[i + 1] not in '.,;':
                    res.append(t[0])
            elif isinstance(t, ParseResults):
                res += t.asList()
            elif t != '.':
                res.append(t)
            if DEBUG:
                print(len(res), t)
        if DEBUG:
            import json
            print(json.dumps(res, indent=2))
        return res
    except:
        if DEBUG:
            import traceback
            traceback.print_exc()
        raise