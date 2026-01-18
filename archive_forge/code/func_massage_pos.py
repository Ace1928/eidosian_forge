import itertools
import functools
import importlib.util
def massage_pos(pos, nangles=180, flatten=False):
    """Rotate a position dict's points to cover a small vertical span"""
    import numpy as np
    xy = np.empty((len(pos), 2))
    for i, (x, y) in enumerate(pos.values()):
        xy[i, 0] = x
        xy[i, 1] = y
    thetas = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    rxys = (rotate(xy, theta) for theta in thetas)
    rxy0 = min(rxys, key=lambda rxy: span(rxy))
    if flatten is True:
        flatten = 2
    if flatten:
        rxy0[:, 1] /= flatten
    return dict(zip(pos, rxy0))