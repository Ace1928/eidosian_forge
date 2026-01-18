from fontTools.ttLib import TTFont
from collections import defaultdict
from operator import add
from functools import reduce
def byteCost(widths, default, nominal):
    if not hasattr(widths, 'items'):
        d = defaultdict(int)
        for w in widths:
            d[w] += 1
        widths = d
    cost = 0
    for w, freq in widths.items():
        if w == default:
            continue
        diff = abs(w - nominal)
        if diff <= 107:
            cost += freq
        elif diff <= 1131:
            cost += freq * 2
        else:
            cost += freq * 5
    return cost