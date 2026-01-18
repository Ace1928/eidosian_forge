import re
from typing import Iterable, List, Tuple
from ._loop import loop_last
from .cells import cell_len, chop_cells
def divide_line(text: str, width: int, fold: bool=True) -> List[int]:
    divides: List[int] = []
    append = divides.append
    line_position = 0
    _cell_len = cell_len
    for start, _end, word in words(text):
        word_length = _cell_len(word.rstrip())
        if line_position + word_length > width:
            if word_length > width:
                if fold:
                    chopped_words = chop_cells(word, max_size=width, position=0)
                    for last, line in loop_last(chopped_words):
                        if start:
                            append(start)
                        if last:
                            line_position = _cell_len(line)
                        else:
                            start += len(line)
                else:
                    if start:
                        append(start)
                    line_position = _cell_len(word)
            elif line_position and start:
                append(start)
                line_position = _cell_len(word)
        else:
            line_position += _cell_len(word)
    return divides