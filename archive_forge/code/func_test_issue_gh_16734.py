import threading
from sympy.core.function import Function, UndefinedFunction
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan)
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify  # can't import as S yet
from sympy.core.symbol import uniquely_named_symbol, _symbol, Str
from sympy.testing.pytest import raises, skip_under_pyodide
from sympy.core.symbol import disambiguate
@skip_under_pyodide('Cannot create threads under pyodide.')
def test_issue_gh_16734():
    syms = list(symbols('x, y'))

    def thread1():
        for n in range(1000):
            syms[0], syms[1] = symbols(f'x{n}, y{n}')
            syms[0].is_positive
        syms[0] = None

    def thread2():
        while syms[0] is not None:
            result = syms[0] == syms[1]
    thread = threading.Thread(target=thread1)
    thread.start()
    thread2()
    thread.join()