import math
import types
from itertools import permutations
def categorical_multiedge_match(attr, default):
    if isinstance(attr, str):

        def match(datasets1, datasets2):
            values1 = {data.get(attr, default) for data in datasets1.values()}
            values2 = {data.get(attr, default) for data in datasets2.values()}
            return values1 == values2
    else:
        attrs = list(zip(attr, default))

        def match(datasets1, datasets2):
            values1 = set()
            for data1 in datasets1.values():
                x = tuple((data1.get(attr, d) for attr, d in attrs))
                values1.add(x)
            values2 = set()
            for data2 in datasets2.values():
                x = tuple((data2.get(attr, d) for attr, d in attrs))
                values2.add(x)
            return values1 == values2
    return match