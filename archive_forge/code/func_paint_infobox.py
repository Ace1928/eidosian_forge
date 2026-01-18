import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def paint_infobox(rows, columns, matches, funcprops, arg_pos, match, docstring, config, match_format):
    """Returns painted completions, funcprops, match, docstring etc."""
    if not (rows and columns):
        return FSArray(0, 0)
    width = columns - 4
    from_argspec = formatted_argspec(funcprops, arg_pos, width, config) if funcprops else []
    from_doc = formatted_docstring(docstring, width, config) if docstring else []
    from_matches = matches_lines(max(1, rows - len(from_argspec) - 2), width, matches, match, config, match_format) if matches else []
    lines = from_argspec + from_matches + from_doc

    def add_border(line):
        """Add colored borders left and right to a line."""
        new_line = border_color(config.left_border + ' ')
        new_line += line.ljust(width)[:width]
        new_line += border_color(' ' + config.right_border)
        return new_line
    border_color = func_for_letter(config.color_scheme['main'])
    top_line = border_color(config.left_top_corner + config.top_border * (width + 2) + config.right_top_corner)
    bottom_line = border_color(config.left_bottom_corner + config.bottom_border * (width + 2) + config.right_bottom_corner)
    output_lines = list(itertools.chain((top_line,), map(add_border, lines), (bottom_line,)))
    r = fsarray(output_lines[:min(rows - 1, len(output_lines) - 1)] + output_lines[-1:])
    return r