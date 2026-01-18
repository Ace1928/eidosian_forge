from sympy.matrices.dense import eye, Matrix
from sympy.tensor.tensor import tensor_indices, TensorHead, tensor_heads, \
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex, \
from sympy import Symbol
def test_kahane_algorithm():

    def tfunc(e):
        return _simplify_single_line(e)
    execute_gamma_simplify_tests_for_function(tfunc, D=4)