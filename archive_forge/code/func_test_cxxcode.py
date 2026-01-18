from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p
def test_cxxcode():
    assert sorted(cxxcode(sqrt(x) * 0.5).split('*')) == sorted(['0.5', 'std::sqrt(x)'])