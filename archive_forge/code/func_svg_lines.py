from __future__ import annotations
import math
import re
import numpy as np
def svg_lines(x1, y1, x2, y2, max_n=20):
    """Convert points into lines of text for an SVG plot

    Examples
    --------
    >>> svg_lines([0, 1], [0, 0], [10, 11], [1, 1])  # doctest: +NORMALIZE_WHITESPACE
    ['  <line x1="0" y1="0" x2="10" y2="1" style="stroke-width:2" />',
     '  <line x1="1" y1="0" x2="11" y2="1" style="stroke-width:2" />']
    """
    n = len(x1)
    if n > max_n:
        indices = np.linspace(0, n - 1, max_n, dtype='int')
    else:
        indices = range(n)
    lines = ['  <line x1="%d" y1="%d" x2="%d" y2="%d" />' % (x1[i], y1[i], x2[i], y2[i]) for i in indices]
    lines[0] = lines[0].replace(' /', ' style="stroke-width:2" /')
    lines[-1] = lines[-1].replace(' /', ' style="stroke-width:2" /')
    return lines