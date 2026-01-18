import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def parse_line_ranges(line_ranges: Sequence[str]) -> List[Tuple[int, int]]:
    lines: List[Tuple[int, int]] = []
    for lines_str in line_ranges:
        parts = lines_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Incorrect --line-ranges format, expect 'START-END', found {lines_str!r}")
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            raise ValueError(f'Incorrect --line-ranges value, expect integer ranges, found {lines_str!r}') from None
        else:
            lines.append((start, end))
    return lines