from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p
def test_cxxcode_nested_minmax():
    assert cxxcode(Max(Min(x, y), Min(u, v))) == 'std::max(std::min(u, v), std::min(x, y))'
    assert cxxcode(Min(Max(x, y), Max(u, v))) == 'std::min(std::max(u, v), std::max(x, y))'