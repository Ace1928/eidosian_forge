import collections
import timeit
import tree
def map_to_list(func, *args):
    return list(map(func, *args))