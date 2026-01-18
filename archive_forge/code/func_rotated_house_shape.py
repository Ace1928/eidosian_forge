import itertools
import functools
import importlib.util
def rotated_house_shape(xy, r=0.4):
    x, y = xy
    return [[x - r, y - r], [x - r, y + r], [x, y + r], [x + r, y], [x, y - r]]