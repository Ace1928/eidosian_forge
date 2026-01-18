import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def convert_one_fmt_off_pair(node: Node, mode: Mode, lines: Collection[Tuple[int, int]]) -> bool:
    """Convert content of a single `# fmt: off`/`# fmt: on` into a standalone comment.

    Returns True if a pair was converted.
    """
    for leaf in node.leaves():
        previous_consumed = 0
        for comment in list_comments(leaf.prefix, is_endmarker=False):
            should_pass_fmt = comment.value in FMT_OFF or _contains_fmt_skip_comment(comment.value, mode)
            if not should_pass_fmt:
                previous_consumed = comment.consumed
                continue
            if should_pass_fmt and comment.type != STANDALONE_COMMENT:
                prev = preceding_leaf(leaf)
                if prev:
                    if comment.value in FMT_OFF and prev.type not in WHITESPACE:
                        continue
                    if _contains_fmt_skip_comment(comment.value, mode) and prev.type in WHITESPACE:
                        continue
            ignored_nodes = list(generate_ignored_nodes(leaf, comment, mode))
            if not ignored_nodes:
                continue
            first = ignored_nodes[0]
            parent = first.parent
            prefix = first.prefix
            if comment.value in FMT_OFF:
                first.prefix = prefix[comment.consumed:]
            if _contains_fmt_skip_comment(comment.value, mode):
                first.prefix = ''
                standalone_comment_prefix = prefix
            else:
                standalone_comment_prefix = prefix[:previous_consumed] + '\n' * comment.newlines
            hidden_value = ''.join((str(n) for n in ignored_nodes))
            comment_lineno = leaf.lineno - comment.newlines
            if comment.value in FMT_OFF:
                fmt_off_prefix = ''
                if len(lines) > 0 and (not any((line[0] <= comment_lineno <= line[1] for line in lines))):
                    fmt_off_prefix = prefix.split(comment.value)[0]
                    if '\n' in fmt_off_prefix:
                        fmt_off_prefix = fmt_off_prefix.split('\n')[-1]
                standalone_comment_prefix += fmt_off_prefix
                hidden_value = comment.value + '\n' + hidden_value
            if _contains_fmt_skip_comment(comment.value, mode):
                hidden_value += (comment.leading_whitespace if Preview.no_normalize_fmt_skip_whitespace in mode else '  ') + comment.value
            if hidden_value.endswith('\n'):
                hidden_value = hidden_value[:-1]
            first_idx: Optional[int] = None
            for ignored in ignored_nodes:
                index = ignored.remove()
                if first_idx is None:
                    first_idx = index
            assert parent is not None, 'INTERNAL ERROR: fmt: on/off handling (1)'
            assert first_idx is not None, 'INTERNAL ERROR: fmt: on/off handling (2)'
            parent.insert_child(first_idx, Leaf(STANDALONE_COMMENT, hidden_value, prefix=standalone_comment_prefix, fmt_pass_converted_first_leaf=first_leaf_of(first)))
            return True
    return False