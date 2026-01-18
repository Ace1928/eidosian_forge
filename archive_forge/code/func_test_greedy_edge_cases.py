import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_greedy_edge_cases():
    expression = 'abc,cfd,dbe,efa'
    dim_dict = {k: 20 for k in expression.replace(',', '')}
    tensors = oe.helpers.build_views(expression, dimension_dict=dim_dict)
    path, path_str = oe.contract_path(expression, *tensors, optimize='greedy', memory_limit='max_input')
    assert check_path(path, [(0, 1, 2, 3)])
    path, path_str = oe.contract_path(expression, *tensors, optimize='greedy', memory_limit=-1)
    assert check_path(path, [(0, 1), (0, 2), (0, 1)])