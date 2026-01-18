import copy
import io
from collections import (
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
from . import (
def generate_row(self, row_data: Sequence[Any], is_header: bool, *, fill_char: str=SPACE, pre_line: str=EMPTY, inter_cell: str=2 * SPACE, post_line: str=EMPTY) -> str:
    """
        Generate a header or data table row

        :param row_data: data with an entry for each column in the row
        :param is_header: True if writing a header cell, otherwise writing a data cell. This determines whether to
                          use header or data alignment settings as well as maximum lines to wrap.
        :param fill_char: character that fills remaining space in a cell. Defaults to space. If this is a tab,
                          then it will be converted to one space. (Cannot be a line breaking character)
        :param pre_line: string to print before each line of a row. This can be used for a left row border and
                         padding before the first cell's text. (Defaults to blank)
        :param inter_cell: string to print where two cells meet. This can be used for a border between cells and padding
                           between it and the 2 cells' text. (Defaults to 2 spaces)
        :param post_line: string to print after each line of a row. This can be used for padding after
                          the last cell's text and a right row border. (Defaults to blank)
        :return: row string
        :raises: ValueError if row_data isn't the same length as self.cols
        :raises: TypeError if fill_char is more than one character (not including ANSI style sequences)
        :raises: ValueError if fill_char, pre_line, inter_cell, or post_line contains an unprintable
                 character like a newline
        """

    class Cell:
        """Inner class which represents a table cell"""

        def __init__(self) -> None:
            self.lines: Deque[str] = deque()
            self.width = 0
    if len(row_data) != len(self.cols):
        raise ValueError('Length of row_data must match length of cols')
    fill_char = fill_char.replace('\t', SPACE)
    pre_line = pre_line.replace('\t', SPACE * self.tab_width)
    inter_cell = inter_cell.replace('\t', SPACE * self.tab_width)
    post_line = post_line.replace('\t', SPACE * self.tab_width)
    if len(ansi.strip_style(fill_char)) != 1:
        raise TypeError('Fill character must be exactly one character long')
    validation_dict = {'fill_char': fill_char, 'pre_line': pre_line, 'inter_cell': inter_cell, 'post_line': post_line}
    for key, val in validation_dict.items():
        if ansi.style_aware_wcswidth(val) == -1:
            raise ValueError(f'{key} contains an unprintable character')
    total_lines = 0
    cells = list()
    for col_index, col in enumerate(self.cols):
        cell = Cell()
        cell.lines, cell.width = self._generate_cell_lines(row_data[col_index], is_header, col, fill_char)
        cells.append(cell)
        total_lines = max(len(cell.lines), total_lines)
    row_buf = io.StringIO()
    for cell_index, cell in enumerate(cells):
        col = self.cols[cell_index]
        vert_align = col.header_vert_align if is_header else col.data_vert_align
        line_diff = total_lines - len(cell.lines)
        if line_diff == 0:
            continue
        padding_line = utils.align_left(EMPTY, fill_char=fill_char, width=cell.width)
        if vert_align == VerticalAlignment.TOP:
            to_top = 0
            to_bottom = line_diff
        elif vert_align == VerticalAlignment.MIDDLE:
            to_top = line_diff // 2
            to_bottom = line_diff - to_top
        else:
            to_top = line_diff
            to_bottom = 0
        for i in range(to_top):
            cell.lines.appendleft(padding_line)
        for i in range(to_bottom):
            cell.lines.append(padding_line)
    for line_index in range(total_lines):
        for cell_index, cell in enumerate(cells):
            if cell_index == 0:
                row_buf.write(pre_line)
            row_buf.write(cell.lines[line_index])
            if cell_index < len(self.cols) - 1:
                row_buf.write(inter_cell)
            if cell_index == len(self.cols) - 1:
                row_buf.write(post_line)
        if line_index < total_lines - 1:
            row_buf.write('\n')
    return row_buf.getvalue()