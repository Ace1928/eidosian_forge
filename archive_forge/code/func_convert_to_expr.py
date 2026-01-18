from sympy.printing import pycode, ccode, fcode
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
def convert_to_expr(self, src_code, mode):
    """Converts the given source code to SymPy Expressions

        Attributes
        ==========

        src_code : String
            the source code or filename of the source code that is to be
            converted

        mode: String
            the mode to determine which parser is to be used according to
            the language of the source code
            f or F for Fortran
            c or C for C/C++

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b) result(r)
        ... integer, intent(in) :: a, b
        ... integer :: x
        ... r = a + b -x
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(r, type=integer, value=0)),
        Declaration(Variable(x, type=integer, value=0)),
        Assignment(Variable(r), a + b - x),
        Return(Variable(r))
        ))]




        """
    if mode.lower() == 'f':
        if lfortran:
            self._expr = src_to_sympy(src_code)
        else:
            raise ImportError('LFortran is not installed, cannot parse Fortran code')
    elif mode.lower() == 'c':
        if cin:
            self._expr = parse_c(src_code)
        else:
            raise ImportError('Clang is not installed, cannot parse C code')
    else:
        raise NotImplementedError('Parser for specified language has not been implemented')