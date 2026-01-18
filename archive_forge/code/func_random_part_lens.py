import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
def random_part_lens(max_n_parts, max_part_size):
    return tuple((random.randint(1, max_part_size) for _ in range(random.randint(1, max_n_parts))))