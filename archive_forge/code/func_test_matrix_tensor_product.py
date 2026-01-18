from sympy.core.random import randint
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, ones, zeros)
from sympy.physics.quantum.matrixutils import (
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_matrix_tensor_product():
    if not np:
        skip('numpy not installed.')
    l1 = zeros(4)
    for i in range(16):
        l1[i] = 2 ** i
    l2 = zeros(4)
    for i in range(16):
        l2[i] = i
    l3 = zeros(2)
    for i in range(4):
        l3[i] = i
    vec = Matrix([1, 2, 3])
    numpyl1 = np.array(l1.tolist())
    numpyl2 = np.array(l2.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, l2]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [l2, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpyl2 = np.array(l3.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, l3]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [l3, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpyl2 = np.array(vec.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, vec]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [vec, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    random_matrix1 = np.random.rand(randint(1, 5), randint(1, 5))
    random_matrix2 = np.random.rand(randint(1, 5), randint(1, 5))
    numpy_product = np.kron(random_matrix1, random_matrix2)
    args = [Matrix(random_matrix1.tolist()), Matrix(random_matrix2.tolist())]
    sympy_product = matrix_tensor_product(*args)
    assert not (sympy_product - Matrix(numpy_product.tolist())).tolist() > (ones(sympy_product.rows, sympy_product.cols) * epsilon).tolist()
    sympy_product = matrix_tensor_product(l1, vec, l2)
    numpy_product = np.kron(l1, np.kron(vec, l2))
    assert numpy_product.tolist() == sympy_product.tolist()