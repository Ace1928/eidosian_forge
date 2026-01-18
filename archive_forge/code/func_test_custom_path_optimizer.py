import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_custom_path_optimizer():

    class NaiveOptimizer(oe.paths.PathOptimizer):

        def __call__(self, inputs, output, size_dict, memory_limit=None):
            self.was_used = True
            return [(0, 1)] * (len(inputs) - 1)
    eq, shapes = oe.helpers.rand_equation(5, 3, seed=42, d_max=3)
    views = list(map(np.ones, shapes))
    exp = oe.contract(eq, *views, optimize=False)
    optimizer = NaiveOptimizer()
    out = oe.contract(eq, *views, optimize=optimizer)
    assert exp == out
    assert optimizer.was_used