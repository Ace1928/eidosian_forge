import difflib
import os
import sys
import textwrap
from typing import Any, Optional, Tuple, Union
def diff_strings(a: str, b: str, fg: Union[str, int]='black', bg: Union[Tuple[str, str], Tuple[int, int]]=('green', 'red'), add_symbols: bool=False) -> str:
    """Compare two strings and return a colored diff with red/green background
    for deletion and insertions.

    a (str): The first string to diff.
    b (str): The second string to diff.
    fg (Union[str, int]): Foreground color. String name or 0 - 256 (see COLORS).
    bg (Union[Tuple[str, str], Tuple[int, int]]): Background colors as
        (insert, delete) tuple of string name or 0 - 256 (see COLORS).
    add_symbols (bool): Whether to add symbols before the diff lines. Uses '+'
        for inserts and '-' for deletions. Default is False.
    RETURNS (str): The formatted diff.
    """
    a_list = a.split('\n')
    b_list = b.split('\n')
    output = []
    matcher = difflib.SequenceMatcher(None, a_list, b_list)
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            for item in a_list[a0:a1]:
                output.append(item)
        if opcode == 'insert' or opcode == 'replace':
            for item in b_list[b0:b1]:
                item = '{} {}'.format(INSERT_SYMBOL, item) if add_symbols else item
                output.append(color(item, fg=fg, bg=bg[0]))
        if opcode == 'delete' or opcode == 'replace':
            for item in a_list[a0:a1]:
                item = '{} {}'.format(DELETE_SYMBOL, item) if add_symbols else item
                output.append(color(item, fg=fg, bg=bg[1]))
    return '\n'.join(output)