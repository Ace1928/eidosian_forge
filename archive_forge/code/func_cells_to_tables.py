import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def cells_to_tables(page, cells) -> list:
    """
    Given a list of bounding boxes (`cells`), return a list of tables that
    hold those cells most simply (and contiguously).
    """

    def bbox_to_corners(bbox) -> tuple:
        x0, top, x1, bottom = bbox
        return ((x0, top), (x0, bottom), (x1, top), (x1, bottom))
    remaining_cells = list(cells)
    current_corners = set()
    current_cells = []
    tables = []
    while len(remaining_cells):
        initial_cell_count = len(current_cells)
        for cell in list(remaining_cells):
            cell_corners = bbox_to_corners(cell)
            if len(current_cells) == 0:
                current_corners |= set(cell_corners)
                current_cells.append(cell)
                remaining_cells.remove(cell)
            else:
                corner_count = sum((c in current_corners for c in cell_corners))
                if corner_count > 0:
                    current_corners |= set(cell_corners)
                    current_cells.append(cell)
                    remaining_cells.remove(cell)
        if len(current_cells) == initial_cell_count:
            tables.append(list(current_cells))
            current_corners.clear()
            current_cells.clear()
    if len(current_cells):
        tables.append(list(current_cells))
    for i in range(len(tables) - 1, -1, -1):
        r = EMPTY_RECT()
        x1_vals = set()
        x0_vals = set()
        for c in tables[i]:
            r |= c
            x1_vals.add(c[2])
            x0_vals.add(c[0])
        if len(x1_vals) < 2 or len(x0_vals) < 2 or white_spaces.issuperset(page.get_textbox(r, textpage=TEXTPAGE)):
            del tables[i]
    _sorted = sorted(tables, key=lambda t: min(((c[1], c[0]) for c in t)))
    return _sorted