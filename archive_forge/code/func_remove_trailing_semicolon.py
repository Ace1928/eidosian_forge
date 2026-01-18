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
def remove_trailing_semicolon(src: str) -> Tuple[str, bool]:
    """Remove trailing semicolon from Jupyter notebook cell.

    For example,

        fig, ax = plt.subplots()
        ax.plot(x_data, y_data);  # plot data

    would become

        fig, ax = plt.subplots()
        ax.plot(x_data, y_data)  # plot data

    Mirrors the logic in `quiet` from `IPython.core.displayhook`, but uses
    ``tokenize_rt`` so that round-tripping works fine.
    """
    from tokenize_rt import reversed_enumerate, src_to_tokens, tokens_to_src
    tokens = src_to_tokens(src)
    trailing_semicolon = False
    for idx, token in reversed_enumerate(tokens):
        if token.name in TOKENS_TO_IGNORE:
            continue
        if token.name == 'OP' and token.src == ';':
            del tokens[idx]
            trailing_semicolon = True
        break
    if not trailing_semicolon:
        return (src, False)
    return (tokens_to_src(tokens), True)