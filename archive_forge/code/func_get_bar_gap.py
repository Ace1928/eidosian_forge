import math
import warnings
import matplotlib.dates
def get_bar_gap(bar_starts, bar_ends, tol=1e-10):
    if len(bar_starts) == len(bar_ends) and len(bar_starts) > 1:
        sides1 = bar_starts[1:]
        sides2 = bar_ends[:-1]
        gaps = [s2 - s1 for s2, s1 in zip(sides1, sides2)]
        gap0 = gaps[0]
        uniform = all([abs(gap0 - gap) < tol for gap in gaps])
        if uniform:
            return gap0