import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def make_edges(page, clip=None, tset=None, add_lines=None):
    global EDGES
    snap_x = tset.snap_x_tolerance
    snap_y = tset.snap_y_tolerance
    lines_strict = tset.vertical_strategy == 'lines_strict' or tset.horizontal_strategy == 'lines_strict'
    page_height = page.rect.height
    doctop_basis = page.number * page_height
    page_number = page.number + 1
    prect = page.rect
    if page.rotation in (90, 270):
        w, h = prect.br
        prect = Rect(0, 0, h, w)
    if clip is not None:
        clip = Rect(clip)
    else:
        clip = prect

    def are_neighbors(r1, r2):
        """Detect whether r1, r2 are neighbors.

        Defined as:
        The minimum distance between points of r1 and points of r2 is not
        larger than some delta.

        This check supports empty rect-likes and thus also lines.

        Note:
        This type of check is MUCH faster than native Rect containment checks.
        """
        if (r2.x0 - snap_x <= r1.x0 <= r2.x1 + snap_x or r2.x0 - snap_x <= r1.x1 <= r2.x1 + snap_x) and (r2.y0 - snap_y <= r1.y0 <= r2.y1 + snap_y or r2.y0 - snap_y <= r1.y1 <= r2.y1 + snap_y):
            return True
        if (r1.x0 - snap_x <= r2.x0 <= r1.x1 + snap_x or r1.x0 - snap_x <= r2.x1 <= r1.x1 + snap_x) and (r1.y0 - snap_y <= r2.y0 <= r1.y1 + snap_y or r1.y0 - snap_y <= r2.y1 <= r1.y1 + snap_y):
            return True
        return False

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
    bboxes, paths = clean_graphics()

    def is_parallel(p1, p2):
        """Check if line is roughly axis-parallel."""
        if abs(p1.x - p2.x) <= snap_x or abs(p1.y - p2.y) <= snap_y:
            return True
        return False

    def make_line(p, p1, p2, clip):
        """Given 2 points, make a line dictionary for table detection."""
        if not is_parallel(p1, p2):
            return {}
        x0 = min(p1.x, p2.x)
        x1 = max(p1.x, p2.x)
        y0 = min(p1.y, p2.y)
        y1 = max(p1.y, p2.y)
        if x0 > clip.x1 or x1 < clip.x0 or y0 > clip.y1 or (y1 < clip.y0):
            return {}
        if x0 < clip.x0:
            x0 = clip.x0
        if x1 > clip.x1:
            x1 = clip.x1
        if y0 < clip.y0:
            y0 = clip.y0
        if y1 > clip.y1:
            y1 = clip.y1
        width = x1 - x0
        height = y1 - y0
        if width == height == 0:
            return {}
        line_dict = {'x0': x0, 'y0': page_height - y0, 'x1': x1, 'y1': page_height - y1, 'width': width, 'height': height, 'pts': [(x0, y0), (x1, y1)], 'linewidth': p['width'], 'stroke': True, 'fill': False, 'evenodd': False, 'stroking_color': p['color'] if p['color'] else p['fill'], 'non_stroking_color': None, 'object_type': 'line', 'page_number': page_number, 'stroking_pattern': None, 'non_stroking_pattern': None, 'top': y0, 'bottom': y1, 'doctop': y0 + doctop_basis}
        return line_dict
    for p in paths:
        items = p['items']
        if p['closePath'] and items[0][0] == 'l' and (items[-1][0] == 'l'):
            items.append(('l', items[-1][2], items[0][1]))
        for i in items:
            if i[0] not in ('l', 're', 'qu'):
                continue
            if i[0] == 'l':
                p1, p2 = i[1:]
                line_dict = make_line(p, p1, p2, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
            elif i[0] == 're':
                rect = i[1].normalize()
                if rect.height <= snap_y and rect.width <= snap_x:
                    continue
                if rect.width <= snap_x:
                    x = abs(rect.x1 + rect.x0) / 2
                    p1 = Point(x, rect.y0)
                    p2 = Point(x, rect.y1)
                    line_dict = make_line(p, p1, p2, clip)
                    if line_dict:
                        EDGES.append(line_to_edge(line_dict))
                    continue
                if rect.height <= snap_y:
                    y = abs(rect.y1 + rect.y0) / 2
                    p1 = Point(rect.x0, y)
                    p2 = Point(rect.x1, y)
                    line_dict = make_line(p, p1, p2, clip)
                    if line_dict:
                        EDGES.append(line_to_edge(line_dict))
                    continue
                line_dict = make_line(p, rect.tl, rect.bl, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
                line_dict = make_line(p, rect.bl, rect.br, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
                line_dict = make_line(p, rect.br, rect.tr, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
                line_dict = make_line(p, rect.tr, rect.tl, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
            else:
                ul, ur, ll, lr = i[1]
                line_dict = make_line(p, ul, ll, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
                line_dict = make_line(p, ll, lr, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
                line_dict = make_line(p, lr, ur, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
                line_dict = make_line(p, ur, ul, clip)
                if line_dict:
                    EDGES.append(line_to_edge(line_dict))
    path = {'color': (0, 0, 0), 'fill': None, 'width': 1}
    for bbox in bboxes:
        line_dict = make_line(path, bbox.tl, bbox.tr, clip)
        if line_dict:
            EDGES.append(line_to_edge(line_dict))
        line_dict = make_line(path, bbox.bl, bbox.br, clip)
        if line_dict:
            EDGES.append(line_to_edge(line_dict))
        line_dict = make_line(path, bbox.tl, bbox.bl, clip)
        if line_dict:
            EDGES.append(line_to_edge(line_dict))
        line_dict = make_line(path, bbox.tr, bbox.br, clip)
        if line_dict:
            EDGES.append(line_to_edge(line_dict))
    if add_lines is not None:
        assert isinstance(add_lines, (tuple, list))
    else:
        add_lines = []
    for p1, p2 in add_lines:
        p1 = Point(p1)
        p2 = Point(p2)
        line_dict = make_line(path, p1, p2, clip)
        if line_dict:
            EDGES.append(line_to_edge(line_dict))