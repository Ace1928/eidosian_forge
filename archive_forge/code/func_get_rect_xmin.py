import math
import warnings
import matplotlib.dates
def get_rect_xmin(data):
    """Find minimum x value from four (x,y) vertices."""
    return min(data[0][0], data[1][0], data[2][0], data[3][0])