from collections import namedtuple
from math import floor, ceil
def clip_line(line, b_box, use_float=False):
    """Algorithm to calculate the clipped line.

    We calculate the coordinates of the part of the line segment within the
    bounding box (defined by left, top, right, bottom). The we write
    the coordinates of the line segment into "line", much like the C-algorithm.
    With `use_float` True, clip_line is usable for float-clipping.

    Returns: true if the line segment cuts the bounding box (false otherwise)
    """

    def inside(code):
        return not code

    def accept(code_a, code_b):
        return not (code_a or code_b)

    def reject(code_a, code_b):
        return code_a and code_b
    assert isinstance(line, list)
    x_1, y_1, x_2, y_2 = line
    dtype = float if use_float else int
    while True:
        code1 = encode((x_1, y_1), b_box)
        code2 = encode((x_2, y_2), b_box)
        if accept(code1, code2):
            line[:] = (x_1, y_1, x_2, y_2)
            return True
        if reject(code1, code2):
            return False
        if inside(code1):
            x_1, x_2 = (x_2, x_1)
            y_1, y_2 = (y_2, y_1)
            code1, code2 = (code2, code1)
        slope = (y_2 - y_1) / float(x_2 - x_1) if x_2 != x_1 else 1.0
        if code1 & LEFT_EDGE:
            y_1 += dtype((b_box.left - x_1) * slope)
            x_1 = b_box.left
        elif code1 & RIGHT_EDGE:
            y_1 += dtype((b_box.right - x_1) * slope)
            x_1 = b_box.right
        elif code1 & BOTTOM_EDGE:
            if x_2 != x_1:
                x_1 += dtype((b_box.bottom - y_1) / slope)
            y_1 = b_box.bottom
        elif code1 & TOP_EDGE:
            if x_2 != x_1:
                x_1 += dtype((b_box.top - y_1) / slope)
            y_1 = b_box.top