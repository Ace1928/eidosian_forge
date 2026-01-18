from .links_base import Strand, Crossing, Link
import random
import collections
def remove_space(point_dict, i):
    """
    Remove the points i and i + 1
    """
    ans = {}
    for a, v in point_dict.items():
        if a < i:
            ans[a] = v
        if a > i + 1:
            ans[a - 2] = v
    return ans