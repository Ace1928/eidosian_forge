from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p
def test_CXX11CodePrinter():
    assert CXX11CodePrinter().doprint(log1p(x)) == 'std::log1p(x)'
    cxx11printer = CXX11CodePrinter()
    assert cxx11printer.language == 'C++'
    assert cxx11printer.standard == 'C++11'
    assert 'operator' in cxx11printer.reserved_words
    assert 'noexcept' in cxx11printer.reserved_words
    assert 'concept' not in cxx11printer.reserved_words