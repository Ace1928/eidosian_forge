import math
import types
from itertools import permutations
def categorical_node_match(attr, default):
    if isinstance(attr, str):

        def match(data1, data2):
            return data1.get(attr, default) == data2.get(attr, default)
    else:
        attrs = list(zip(attr, default))

        def match(data1, data2):
            return all((data1.get(attr, d) == data2.get(attr, d) for attr, d in attrs))
    return match