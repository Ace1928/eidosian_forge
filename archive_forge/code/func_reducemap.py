from functools import reduce
from operator import mul, add
def reducemap(args, reduce_op, map_op):
    return reduce(reduce_op, map(map_op, *args))