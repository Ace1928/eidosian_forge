import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
def parsekeywordpairs(signature: str) -> Dict[str, str]:
    preamble = True
    stack = []
    substack: List[str] = []
    parendepth = 0
    annotation = False
    for token, value in Python3Lexer().get_tokens(signature):
        if preamble:
            if token is Token.Punctuation and value == '(':
                preamble = False
            continue
        if token is Token.Punctuation:
            if value in '({[':
                parendepth += 1
            elif value in ')}]':
                parendepth -= 1
            elif value == ':':
                if parendepth == -1:
                    break
                elif parendepth == 0:
                    annotation = True
            if (value, parendepth) in ((',', 0), (')', -1)):
                stack.append(substack)
                substack = []
                annotation = False
                continue
        elif token is Token.Operator and value == '=' and (parendepth == 0):
            annotation = False
        if value and (not annotation) and (parendepth > 0 or value.strip()):
            substack.append(value)
    return {item[0]: ''.join(item[2:]) for item in stack if len(item) >= 3}