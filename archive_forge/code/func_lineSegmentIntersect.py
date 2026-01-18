from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def lineSegmentIntersect(xxx_todo_changeme2, xxx_todo_changeme3, xxx_todo_changeme4, xxx_todo_changeme5):
    x00, y00 = xxx_todo_changeme2
    x01, y01 = xxx_todo_changeme3
    x10, y10 = xxx_todo_changeme4
    x11, y11 = xxx_todo_changeme5
    p = (x00, y00)
    r = (x01 - x00, y01 - y00)
    q = (x10, y10)
    s = (x11 - x10, y11 - y10)
    rs = float(r[0] * s[1] - r[1] * s[0])
    qp = (q[0] - p[0], q[1] - p[1])
    qpr = qp[0] * r[1] - qp[1] * r[0]
    qps = qp[0] * s[1] - qp[1] * s[0]
    if abs(rs) < 1e-08:
        if abs(qpr) < 1e-08:
            return 'collinear'
        return None
    t = qps / rs
    u = qpr / rs
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p[0] + t * r[0], p[1] + t * r[1])