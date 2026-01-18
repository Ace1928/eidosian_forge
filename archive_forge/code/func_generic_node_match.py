import math
import types
from itertools import permutations
def generic_node_match(attr, default, op):
    if isinstance(attr, str):

        def match(data1, data2):
            return op(data1.get(attr, default), data2.get(attr, default))
    else:
        attrs = list(zip(attr, default, op))

        def match(data1, data2):
            for attr, d, operator in attrs:
                if not operator(data1.get(attr, d), data2.get(attr, d)):
                    return False
            else:
                return True
    return match