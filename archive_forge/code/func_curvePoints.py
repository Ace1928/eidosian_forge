from rdkit.sping.colors import *
def curvePoints(self, x1, y1, x2, y2, x3, y3, x4, y4):
    """Return a list of points approximating the given Bezier curve."""
    bezierSteps = min(max(max(x1, x2, x3, x4) - min(x1, x2, x3, x3), max(y1, y2, y3, y4) - min(y1, y2, y3, y4)), 200)
    dt1 = 1.0 / bezierSteps
    dt2 = dt1 * dt1
    dt3 = dt2 * dt1
    xx = x1
    yy = y1
    ux = uy = vx = vy = 0
    ax = x4 - 3 * x3 + 3 * x2 - x1
    ay = y4 - 3 * y3 + 3 * y2 - y1
    bx = 3 * x3 - 6 * x2 + 3 * x1
    by = 3 * y3 - 6 * y2 + 3 * y1
    cx = 3 * x2 - 3 * x1
    cy = 3 * y2 - 3 * y1
    mx1 = ax * dt3
    my1 = ay * dt3
    lx1 = bx * dt2
    ly1 = by * dt2
    kx = mx1 + lx1 + cx * dt1
    ky = my1 + ly1 + cy * dt1
    mx = 6 * mx1
    my = 6 * my1
    lx = mx + 2 * lx1
    ly = my + 2 * ly1
    pointList = [(xx, yy)]
    for i in range(bezierSteps):
        xx = xx + ux + kx
        yy = yy + uy + ky
        ux = ux + vx + lx
        uy = uy + vy + ly
        vx = vx + mx
        vy = vy + my
        pointList.append((xx, yy))
    return pointList