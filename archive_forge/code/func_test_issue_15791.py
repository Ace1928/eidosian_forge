from sympy.printing.codeprinter import CodePrinter
from sympy.core import symbols
from sympy.core.symbol import Dummy
from sympy.testing.pytest import raises
def test_issue_15791():

    class CrashingCodePrinter(CodePrinter):

        def emptyPrinter(self, obj):
            raise NotImplementedError
    from sympy.matrices import MutableSparseMatrix, ImmutableSparseMatrix
    c = CrashingCodePrinter()
    with raises(NotImplementedError):
        c.doprint(ImmutableSparseMatrix(2, 2, {}))
    with raises(NotImplementedError):
        c.doprint(MutableSparseMatrix(2, 2, {}))