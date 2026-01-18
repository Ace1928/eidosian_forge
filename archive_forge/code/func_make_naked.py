import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def make_naked(string: str, string_prefix: str) -> str:
    """Strip @string (i.e. make it a "naked" string)

            Pre-conditions:
                * assert_is_leaf_string(@string)

            Returns:
                A string that is identical to @string except that
                @string_prefix has been stripped, the surrounding QUOTE
                characters have been removed, and any remaining QUOTE
                characters have been escaped.
            """
    assert_is_leaf_string(string)
    if 'f' in string_prefix:
        f_expressions = (string[span[0] + 1:span[1] - 1] for span in iter_fexpr_spans(string))
        debug_expressions_contain_visible_quotes = any((re.search('.*[\\\'\\"].*(?<![!:=])={1}(?!=)(?![^\\s:])', expression) for expression in f_expressions))
        if not debug_expressions_contain_visible_quotes:
            string = _toggle_fexpr_quotes(string, QUOTE)
    RE_EVEN_BACKSLASHES = '(?:(?<!\\\\)(?:\\\\\\\\)*)'
    naked_string = string[len(string_prefix) + 1:-1]
    naked_string = re.sub('(' + RE_EVEN_BACKSLASHES + ')' + QUOTE, '\\1\\\\' + QUOTE, naked_string)
    return naked_string