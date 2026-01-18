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
def mask_cell(src: str) -> Tuple[str, List[Replacement]]:
    """Mask IPython magics so content becomes parseable Python code.

    For example,

        %matplotlib inline
        'foo'

    becomes

        "25716f358c32750e"
        'foo'

    The replacements are returned, along with the transformed code.
    """
    replacements: List[Replacement] = []
    try:
        ast.parse(src)
    except SyntaxError:
        pass
    else:
        return (src, replacements)
    from IPython.core.inputtransformer2 import TransformerManager
    transformer_manager = TransformerManager()
    transformed = transformer_manager.transform_cell(src)
    transformed, cell_magic_replacements = replace_cell_magics(transformed)
    replacements += cell_magic_replacements
    transformed = transformer_manager.transform_cell(transformed)
    transformed, magic_replacements = replace_magics(transformed)
    if len(transformed.splitlines()) != len(src.splitlines()):
        raise NothingChanged
    replacements += magic_replacements
    return (transformed, replacements)