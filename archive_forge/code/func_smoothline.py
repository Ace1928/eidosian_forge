import tkinter
from tkinter.constants import *
def smoothline(document, coords):
    """smoothed polyline"""
    element = document.createElement('path')
    path = []
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    def pt(points):
        x0, y0 = points[0]
        x1, y1 = points[1]
        p0 = (2 * x0 - x1, 2 * y0 - y1)
        x0, y0 = points[-1]
        x1, y1 = points[-2]
        pn = (2 * x0 - x1, 2 * y0 - y1)
        p = [p0] + points[1:-1] + [pn]
        for i in range(1, len(points) - 1):
            a = p[i - 1]
            b = p[i]
            c = p[i + 1]
            yield (lerp(a, b, 0.5), b, lerp(b, c, 0.5))
    for i, (A, B, C) in enumerate(pt(points)):
        if i == 0:
            path.append('M%s,%s Q%s,%s %s,%s' % (A[0], A[1], B[0], B[1], C[0], C[1]))
        else:
            path.append('T%s,%s' % (C[0], C[1]))
    element.setAttribute('d', ' '.join(path))
    return element