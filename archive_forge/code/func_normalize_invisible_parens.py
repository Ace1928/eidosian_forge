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
def normalize_invisible_parens(node: Node, parens_after: Set[str], *, mode: Mode, features: Collection[Feature]) -> None:
    """Make existing optional parentheses invisible or create new ones.

    `parens_after` is a set of string leaf values immediately after which parens
    should be put.

    Standardizes on visible parentheses for single-element tuples, and keeps
    existing visible parentheses for other tuples and generator expressions.
    """
    for pc in list_comments(node.prefix, is_endmarker=False):
        if pc.value in FMT_OFF:
            return
    if node.type == syms.with_stmt:
        _maybe_wrap_cms_in_parens(node, mode, features)
    check_lpar = False
    for index, child in enumerate(list(node.children)):
        if isinstance(child, Node) and child.type == syms.annassign:
            normalize_invisible_parens(child, parens_after=parens_after, mode=mode, features=features)
        if isinstance(child, Node) and child.type == syms.case_block:
            normalize_invisible_parens(child, parens_after={'case'}, mode=mode, features=features)
        if isinstance(child, Node) and child.type == syms.guard and (Preview.parens_for_long_if_clauses_in_case_block in mode):
            normalize_invisible_parens(child, parens_after={'if'}, mode=mode, features=features)
        if index == 0 and isinstance(child, Node) and (child.type == syms.testlist_star_expr):
            check_lpar = True
        if check_lpar:
            if child.type == syms.atom and node.type == syms.for_stmt and isinstance(child.prev_sibling, Leaf) and (child.prev_sibling.type == token.NAME) and (child.prev_sibling.value == 'for'):
                if maybe_make_parens_invisible_in_atom(child, parent=node, remove_brackets_around_comma=True):
                    wrap_in_parentheses(node, child, visible=False)
            elif isinstance(child, Node) and node.type == syms.with_stmt:
                remove_with_parens(child, node)
            elif child.type == syms.atom:
                if maybe_make_parens_invisible_in_atom(child, parent=node):
                    wrap_in_parentheses(node, child, visible=False)
            elif is_one_tuple(child):
                wrap_in_parentheses(node, child, visible=True)
            elif node.type == syms.import_from:
                _normalize_import_from(node, child, index)
                break
            elif index == 1 and child.type == token.STAR and (node.type == syms.except_clause):
                continue
            elif isinstance(child, Leaf) and child.next_sibling is not None and (child.next_sibling.type == token.COLON) and (child.value == 'case'):
                break
            elif not (isinstance(child, Leaf) and is_multiline_string(child)):
                wrap_in_parentheses(node, child, visible=False)
        comma_check = child.type == token.COMMA
        check_lpar = isinstance(child, Leaf) and (child.value in parens_after or comma_check)