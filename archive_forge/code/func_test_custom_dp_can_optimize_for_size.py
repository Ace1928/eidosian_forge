import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_custom_dp_can_optimize_for_size():
    eq, shapes = oe.helpers.rand_equation(10, 4, seed=43)
    opt1 = oe.DynamicProgramming(minimize='flops')
    opt2 = oe.DynamicProgramming(minimize='size')
    info1 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt1)[1]
    info2 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt2)[1]
    assert info1.opt_cost < info2.opt_cost
    assert info1.largest_intermediate > info2.largest_intermediate