import collections
import timeit
import tree
def benchmark_map(map_fn, structure):

    def benchmark_fn():
        return map_fn(lambda v: v, structure)
    return benchmark_fn