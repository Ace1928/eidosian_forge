from functools import total_ordering
@staticmethod
def gen_counter():
    counter = OrderedType.creation_counter
    OrderedType.creation_counter += 1
    return counter