import numpy as np
import pytest
from opt_einsum import contract, contract_path
def test_ellipse_input3():
    string = '...a->...a'
    views = build_views(string)
    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 0], [Ellipsis, 0])
    assert np.allclose(ein, opt)