import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def convert_unchanged_lines(src_node: Node, lines: Collection[Tuple[int, int]]) -> None:
    """Converts unchanged lines to STANDALONE_COMMENT.

    The idea is similar to how `# fmt: on/off` is implemented. It also converts the
    nodes between those markers as a single `STANDALONE_COMMENT` leaf node with
    the unformatted code as its value. `STANDALONE_COMMENT` is a "fake" token
    that will be formatted as-is with its prefix normalized.

    Here we perform two passes:

    1. Visit the top-level statements, and convert them to a single
       `STANDALONE_COMMENT` when unchanged. This speeds up formatting when some
       of the top-level statements aren't changed.
    2. Convert unchanged "unwrapped lines" to `STANDALONE_COMMENT` nodes line by
       line. "unwrapped lines" are divided by the `NEWLINE` token. e.g. a
       multi-line statement is *one* "unwrapped line" that ends with `NEWLINE`,
       even though this statement itself can span multiple lines, and the
       tokenizer only sees the last '
' as the `NEWLINE` token.

    NOTE: During pass (2), comment prefixes and indentations are ALWAYS
    normalized even when the lines aren't changed. This is fixable by moving
    more formatting to pass (1). However, it's hard to get it correct when
    incorrect indentations are used. So we defer this to future optimizations.
    """
    lines_set: Set[int] = set()
    for start, end in lines:
        lines_set.update(range(start, end + 1))
    visitor = _TopLevelStatementsVisitor(lines_set)
    _ = list(visitor.visit(src_node))
    _convert_unchanged_line_by_line(src_node, lines_set)