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
def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
    LL = line.leaves
    assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
    string_idx = string_indices[0]
    is_valid_index = is_valid_index_factory(LL)
    insert_str_child = insert_str_child_factory(LL[string_idx])
    comma_idx = -1
    ends_with_comma = False
    if LL[comma_idx].type == token.COMMA:
        ends_with_comma = True
    leaves_to_steal_comments_from = [LL[string_idx]]
    if ends_with_comma:
        leaves_to_steal_comments_from.append(LL[comma_idx])
    first_line = line.clone()
    left_leaves = LL[:string_idx]
    old_parens_exist = False
    if left_leaves and left_leaves[-1].type == token.LPAR:
        old_parens_exist = True
        leaves_to_steal_comments_from.append(left_leaves[-1])
        left_leaves.pop()
    append_leaves(first_line, line, left_leaves)
    lpar_leaf = Leaf(token.LPAR, '(')
    if old_parens_exist:
        replace_child(LL[string_idx - 1], lpar_leaf)
    else:
        insert_str_child(lpar_leaf)
    first_line.append(lpar_leaf)
    for leaf in leaves_to_steal_comments_from:
        for comment_leaf in line.comments_after(leaf):
            first_line.append(comment_leaf, preformatted=True)
    yield Ok(first_line)
    string_value = LL[string_idx].value
    string_line = Line(mode=line.mode, depth=line.depth + 1, inside_brackets=True, should_split_rhs=line.should_split_rhs, magic_trailing_comma=line.magic_trailing_comma)
    string_leaf = Leaf(token.STRING, string_value)
    insert_str_child(string_leaf)
    string_line.append(string_leaf)
    old_rpar_leaf = None
    if is_valid_index(string_idx + 1):
        right_leaves = LL[string_idx + 1:]
        if ends_with_comma:
            right_leaves.pop()
        if old_parens_exist:
            assert right_leaves and right_leaves[-1].type == token.RPAR, f'Apparently, old parentheses do NOT exist?! (left_leaves={left_leaves}, right_leaves={right_leaves})'
            old_rpar_leaf = right_leaves.pop()
        elif right_leaves and right_leaves[-1].type == token.RPAR:
            opening_bracket = right_leaves[-1].opening_bracket
            if opening_bracket is not None and opening_bracket in left_leaves:
                index = left_leaves.index(opening_bracket)
                if 0 < index < len(left_leaves) - 1 and left_leaves[index - 1].type == token.COLON and (left_leaves[index + 1].value == 'lambda'):
                    right_leaves.pop()
        append_leaves(string_line, line, right_leaves)
    yield Ok(string_line)
    last_line = line.clone()
    last_line.bracket_tracker = first_line.bracket_tracker
    new_rpar_leaf = Leaf(token.RPAR, ')')
    if old_rpar_leaf is not None:
        replace_child(old_rpar_leaf, new_rpar_leaf)
    else:
        insert_str_child(new_rpar_leaf)
    last_line.append(new_rpar_leaf)
    if ends_with_comma:
        comma_leaf = Leaf(token.COMMA, ',')
        replace_child(LL[comma_idx], comma_leaf)
        last_line.append(comma_leaf)
    yield Ok(last_line)