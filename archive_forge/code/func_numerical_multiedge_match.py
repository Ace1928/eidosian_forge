import math
import types
from itertools import permutations
def numerical_multiedge_match(attr, default, rtol=1e-05, atol=1e-08):
    if isinstance(attr, str):

        def match(datasets1, datasets2):
            values1 = sorted((data.get(attr, default) for data in datasets1.values()))
            values2 = sorted((data.get(attr, default) for data in datasets2.values()))
            return allclose(values1, values2, rtol=rtol, atol=atol)
    else:
        attrs = list(zip(attr, default))

        def match(datasets1, datasets2):
            values1 = []
            for data1 in datasets1.values():
                x = tuple((data1.get(attr, d) for attr, d in attrs))
                values1.append(x)
            values2 = []
            for data2 in datasets2.values():
                x = tuple((data2.get(attr, d) for attr, d in attrs))
                values2.append(x)
            values1.sort()
            values2.sort()
            for xi, yi in zip(values1, values2):
                if not allclose(xi, yi, rtol=rtol, atol=atol):
                    return False
            else:
                return True
    return match