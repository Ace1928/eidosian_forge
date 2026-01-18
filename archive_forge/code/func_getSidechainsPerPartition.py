import itertools
import math
import operator
import random
from functools import reduce
def getSidechainsPerPartition(self, total_num_partitions_per_rgroup):
    """total_num_partitions -> [num_fragments/partition for rgroup1, 
                                    num_fragments/partition for rgroup2]
        return the number of sidechains in a partition
        for each rgroup"""
    sizes = [(libIdx, max(rg.count() // total_num_partitions_per_rgroup, 1)) for libIdx, rg in enumerate(self.rgroups)]
    sizes.sort(key=lambda sz: sz[1])
    last_size = 1
    opt_sizes = []
    for libIdx, current_size in sizes[:-1]:
        opt_sizes.append((libIdx, current_size - current_size % last_size))
        last_size = current_size
    libIdx, current_size = sizes[-1]
    opt_sizes.append((libIdx, last_size))
    opt_sizes.sort()
    res = [size for libIdx, size in opt_sizes]
    return res