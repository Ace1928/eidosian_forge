import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
def put_trailing_semicolon_back(src: str, has_trailing_semicolon: bool) -> str:
    """Put trailing semicolon back if cell originally had it.

    Mirrors the logic in `quiet` from `IPython.core.displayhook`, but uses
    ``tokenize_rt`` so that round-tripping works fine.
    """
    if not has_trailing_semicolon:
        return src
    from tokenize_rt import reversed_enumerate, src_to_tokens, tokens_to_src
    tokens = src_to_tokens(src)
    for idx, token in reversed_enumerate(tokens):
        if token.name in TOKENS_TO_IGNORE:
            continue
        tokens[idx] = token._replace(src=token.src + ';')
        break
    else:
        raise AssertionError('INTERNAL ERROR: Was not able to reinstate trailing semicolon. Please report a bug on https://github.com/psf/black/issues.  ') from None
    return str(tokens_to_src(tokens))