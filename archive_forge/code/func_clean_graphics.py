import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def clean_graphics():
    """Detect and join rectangles of "connected" vector graphics."""
    paths = []
    for p in page.get_drawings():
        if p['type'] == 'f' and lines_strict and (p['rect'].width > snap_x) and (p['rect'].height > snap_y):
            continue
        paths.append(p)
    prects = sorted(set([p['rect'] for p in paths]), key=lambda r: (r.y1, r.x0))
    new_rects = []
    while prects:
        prect0 = prects[0]
        repeat = True
        while repeat:
            repeat = False
            for i in range(len(prects) - 1, 0, -1):
                if are_neighbors(prect0, prects[i]):
                    prect0 |= prects[i].tl
                    prect0 |= prects[i].br
                    del prects[i]
                    repeat = True
        if not white_spaces.issuperset(page.get_textbox(prect0, textpage=TEXTPAGE)):
            new_rects.append(prect0)
        del prects[0]
    return (new_rects, paths)