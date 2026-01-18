from __future__ import annotations
import math
import re
import numpy as np
def svg_grid(x, y, offset=(0, 0), skew=(0, 0), size=200):
    """Create lines of SVG text that show a grid

    Parameters
    ----------
    x: numpy.ndarray
    y: numpy.ndarray
    offset: tuple
        translational displacement of the grid in SVG coordinates
    skew: tuple
    """
    x1 = np.zeros_like(y) + offset[0]
    y1 = y + offset[1]
    x2 = np.full_like(y, x[-1]) + offset[0]
    y2 = y + offset[1]
    if skew[0]:
        y2 += x.max() * skew[0]
    if skew[1]:
        x1 += skew[1] * y
        x2 += skew[1] * y
    min_x = min(x1.min(), x2.min())
    min_y = min(y1.min(), y2.min())
    max_x = max(x1.max(), x2.max())
    max_y = max(y1.max(), y2.max())
    max_n = size // 6
    h_lines = ['', '  <!-- Horizontal lines -->'] + svg_lines(x1, y1, x2, y2, max_n)
    x1 = x + offset[0]
    y1 = np.zeros_like(x) + offset[1]
    x2 = x + offset[0]
    y2 = np.full_like(x, y[-1]) + offset[1]
    if skew[0]:
        y1 += skew[0] * x
        y2 += skew[0] * x
    if skew[1]:
        x2 += skew[1] * y.max()
    v_lines = ['', '  <!-- Vertical lines -->'] + svg_lines(x1, y1, x2, y2, max_n)
    color = 'ECB172' if len(x) < max_n and len(y) < max_n else '8B4903'
    corners = f'{x1[0]},{y1[0]} {x1[-1]},{y1[-1]} {x2[-1]},{y2[-1]} {x2[0]},{y2[0]}'
    rect = ['', '  <!-- Colored Rectangle -->', f'  <polygon points="{corners}" style="fill:#{color}A0;stroke-width:0"/>']
    return (h_lines + v_lines + rect, (min_x, max_x, min_y, max_y))