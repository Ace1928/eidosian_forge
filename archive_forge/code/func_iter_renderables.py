from collections import defaultdict
from itertools import chain
from operator import itemgetter
from typing import Dict, Iterable, List, Optional, Tuple
from .align import Align, AlignMethod
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .constrain import Constrain
from .measure import Measurement
from .padding import Padding, PaddingDimensions
from .table import Table
from .text import TextType
from .jupyter import JupyterMixin
def iter_renderables(column_count: int) -> Iterable[Tuple[int, Optional[RenderableType]]]:
    item_count = len(renderables)
    if self.column_first:
        width_renderables = list(zip(renderable_widths, renderables))
        column_lengths: List[int] = [item_count // column_count] * column_count
        for col_no in range(item_count % column_count):
            column_lengths[col_no] += 1
        row_count = (item_count + column_count - 1) // column_count
        cells = [[-1] * column_count for _ in range(row_count)]
        row = col = 0
        for index in range(item_count):
            cells[row][col] = index
            column_lengths[col] -= 1
            if column_lengths[col]:
                row += 1
            else:
                col += 1
                row = 0
        for index in chain.from_iterable(cells):
            if index == -1:
                break
            yield width_renderables[index]
    else:
        yield from zip(renderable_widths, renderables)
    if item_count % column_count:
        for _ in range(column_count - item_count % column_count):
            yield (0, None)