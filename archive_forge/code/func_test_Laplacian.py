from sympy.vector import CoordSys3D, Gradient, Divergence, Curl, VectorZero, Laplacian
from sympy.printing.repr import srepr
def test_Laplacian():
    assert Laplacian(s3) == Laplacian(R.x ** 2 + R.y ** 2 + R.z ** 2)
    assert Laplacian(v3) == Laplacian(R.x ** 2 * R.i + R.y ** 2 * R.j + R.z ** 2 * R.k)
    assert Laplacian(s3).doit() == 6
    assert Laplacian(v3).doit() == 2 * R.i + 2 * R.j + 2 * R.k
    assert srepr(Laplacian(s3)) == 'Laplacian(Add(Pow(R.x, Integer(2)), Pow(R.y, Integer(2)), Pow(R.z, Integer(2))))'