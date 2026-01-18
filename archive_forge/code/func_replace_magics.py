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
def replace_magics(src: str) -> Tuple[str, List[Replacement]]:
    """Replace magics within body of cell.

    Note that 'src' will already have been processed by IPython's
    TransformerManager().transform_cell.

    Example, this

        get_ipython().run_line_magic('matplotlib', 'inline')
        'foo'

    becomes

        "5e67db56d490fd39"
        'foo'

    The replacement, along with the transformed code, are returned.
    """
    replacements = []
    magic_finder = MagicFinder()
    magic_finder.visit(ast.parse(src))
    new_srcs = []
    for i, line in enumerate(src.splitlines(), start=1):
        if i in magic_finder.magics:
            offsets_and_magics = magic_finder.magics[i]
            if len(offsets_and_magics) != 1:
                raise AssertionError(f'Expecting one magic per line, got: {offsets_and_magics}\nPlease report a bug on https://github.com/psf/black/issues.')
            col_offset, magic = (offsets_and_magics[0].col_offset, offsets_and_magics[0].magic)
            mask = get_token(src, magic)
            replacements.append(Replacement(mask=mask, src=magic))
            line = line[:col_offset] + mask
        new_srcs.append(line)
    return ('\n'.join(new_srcs), replacements)