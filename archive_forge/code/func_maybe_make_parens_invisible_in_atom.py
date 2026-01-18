import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def maybe_make_parens_invisible_in_atom(node: LN, parent: LN, remove_brackets_around_comma: bool=False) -> bool:
    """If it's safe, make the parens in the atom `node` invisible, recursively.
    Additionally, remove repeated, adjacent invisible parens from the atom `node`
    as they are redundant.

    Returns whether the node should itself be wrapped in invisible parentheses.
    """
    if node.type not in (syms.atom, syms.expr) or is_empty_tuple(node) or is_one_tuple(node) or (is_yield(node) and parent.type != syms.expr_stmt) or (not remove_brackets_around_comma and max_delimiter_priority_in_atom(node) >= COMMA_PRIORITY) or is_tuple_containing_walrus(node):
        return False
    if is_walrus_assignment(node):
        if parent.type in [syms.annassign, syms.expr_stmt, syms.assert_stmt, syms.return_stmt, syms.except_clause, syms.funcdef, syms.with_stmt, syms.tname, syms.for_stmt, syms.del_stmt, syms.for_stmt]:
            return False
    first = node.children[0]
    last = node.children[-1]
    if is_lpar_token(first) and is_rpar_token(last):
        middle = node.children[1]
        if not is_type_ignore_comment_string(middle.prefix.strip()):
            first.value = ''
            if first.prefix.strip():
                middle.prefix = first.prefix + middle.prefix
            last.value = ''
        maybe_make_parens_invisible_in_atom(middle, parent=parent, remove_brackets_around_comma=remove_brackets_around_comma)
        if is_atom_with_invisible_parens(middle):
            middle.replace(middle.children[1])
            if middle.children[-1].prefix.strip():
                last.prefix = middle.children[-1].prefix + last.prefix
        return False
    return True