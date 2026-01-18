import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def paint_last_events(rows, columns, names, config):
    if not names:
        return fsarray([])
    width = min(max((len(name) for name in names)), columns - 2)
    output_lines = []
    output_lines.append(config.left_top_corner + config.top_border * width + config.right_top_corner)
    for name in reversed(names[max(0, len(names) - (rows - 2)):]):
        output_lines.append(config.left_border + name[:width].center(width) + config.right_border)
    output_lines.append(config.left_bottom_corner + config.bottom_border * width + config.right_bottom_corner)
    return fsarray(output_lines)