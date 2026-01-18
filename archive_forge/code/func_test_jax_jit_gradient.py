import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.skipif(not found_jax, reason='jax not installed.')
def test_jax_jit_gradient():
    eq = 'ij,jk,kl->'
    shapes = ((2, 3), (3, 4), (4, 2))
    views = [np.random.randn(*s) for s in shapes]
    expr = contract_expression(eq, *shapes)
    x0 = expr(*views)
    jit_expr = jax.jit(expr)
    x1 = jit_expr(*views).item()
    assert x1 == pytest.approx(x0, rel=1e-05)
    grad_expr = jax.jit(jax.grad(lambda views: expr(*views)))
    view_grads = grad_expr(views)
    assert all((v1.shape == v2.shape for v1, v2 in zip(views, view_grads)))
    new_views = [v - 0.001 * dv for v, dv in zip(views, view_grads)]
    x2 = jit_expr(*new_views).item()
    assert x2 < x1