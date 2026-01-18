from __future__ import annotations
import math
import re
import numpy as np
def svg_2d(chunks, offset=(0, 0), skew=(0, 0), size=200, sizes=None):
    shape = tuple(map(sum, chunks))
    sizes = sizes or draw_sizes(shape, size=size)
    y, x = grid_points(chunks, sizes)
    lines, (min_x, max_x, min_y, max_y) = svg_grid(x, y, offset=offset, skew=skew, size=size)
    header = '<svg width="%d" height="%d" style="stroke:rgb(0,0,0);stroke-width:1" >\n' % (max_x + 50, max_y + 50)
    footer = '\n</svg>'
    if shape[0] >= 100:
        rotate = -90
    else:
        rotate = 0
    text = ['', '  <!-- Text -->', '  <text x="%f" y="%f" %s >%d</text>' % (max_x / 2, max_y + 20, text_style, shape[1]), '  <text x="%f" y="%f" %s transform="rotate(%d,%f,%f)">%d</text>' % (max_x + 20, max_y / 2, text_style, rotate, max_x + 20, max_y / 2, shape[0])]
    return header + '\n'.join(lines + text) + footer