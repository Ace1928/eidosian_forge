from __future__ import annotations
import math
import re
import numpy as np
def svg_nd(chunks, size=200):
    if len(chunks) % 3 == 1:
        chunks = ((1,),) + chunks
    shape = tuple(map(sum, chunks))
    sizes = draw_sizes(shape, size=size)
    chunks2 = chunks
    sizes2 = sizes
    out = []
    left = 0
    total_height = 0
    while chunks2:
        n = len(chunks2) % 3 or 3
        o = svg(chunks2[:n], sizes=sizes2[:n], offset=(left, 0))
        chunks2 = chunks2[n:]
        sizes2 = sizes2[n:]
        lines = o.split('\n')
        header = lines[0]
        height = float(re.search('height="(\\d*\\.?\\d*)"', header).groups()[0])
        total_height = max(total_height, height)
        width = float(re.search('width="(\\d*\\.?\\d*)"', header).groups()[0])
        left += width + 10
        o = '\n'.join(lines[1:-1])
        out.append(o)
    header = '<svg width="%d" height="%d" style="stroke:rgb(0,0,0);stroke-width:1" >\n' % (left, total_height)
    footer = '\n</svg>'
    return header + '\n\n'.join(out) + footer