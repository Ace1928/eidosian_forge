from typing import Collection, List
from sys import maxsize
def dedent_block_string_lines(lines: Collection[str]) -> List[str]:
    """Produce the value of a block string from its parsed raw value.

    This function works similar to CoffeeScript's block string,
    Python's docstring trim or Ruby's strip_heredoc.

    It implements the GraphQL spec's BlockStringValue() static algorithm.

    Note that this is very similar to Python's inspect.cleandoc() function.
    The difference is that the latter also expands tabs to spaces and
    removes whitespace at the beginning of the first line. Python also has
    textwrap.dedent() which uses a completely different algorithm.

    For internal use only.
    """
    common_indent = maxsize
    first_non_empty_line = None
    last_non_empty_line = -1
    for i, line in enumerate(lines):
        indent = leading_white_space(line)
        if indent == len(line):
            continue
        if first_non_empty_line is None:
            first_non_empty_line = i
        last_non_empty_line = i
        if i and indent < common_indent:
            common_indent = indent
    if first_non_empty_line is None:
        first_non_empty_line = 0
    return [line[common_indent:] if i else line for i, line in enumerate(lines)][first_non_empty_line:last_non_empty_line + 1]