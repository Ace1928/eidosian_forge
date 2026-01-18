import math
from .functions import defun
def aux_J_needed(ctx, xA, xeps4, a, xB1, xM):
    h1 = xeps4 / (632 * xA)
    h2 = xB1 * a * 126.3133741952926
    h2 = h1 * ctx.power(h2 / xM ** 2, (xM - 1) / 3) / xM
    h3 = min(h1, h2)
    return h3