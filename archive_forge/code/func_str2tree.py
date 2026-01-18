import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings
def str2tree(self, exprstr):
    """Converts an expression string to a tree.

        Explanation
        ===========

        Functions are represented by ('func_name(', tree_of_arguments).
        Other expressions are (head_string, mid_tree, tail_str).
        Expressions that do not contain functions are directly returned.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy import Integral, sin
        >>> from sympy.plotting.experimental_lambdify import Lambdifier
        >>> str2tree = Lambdifier([x], x).str2tree

        >>> str2tree(str(Integral(x, (x, 1, y))))
        ('', ('Integral(', 'x, (x, 1, y)'), ')')
        >>> str2tree(str(x+y))
        'x + y'
        >>> str2tree(str(x+y*sin(z)+1))
        ('x + y*', ('sin(', 'z'), ') + 1')
        >>> str2tree('sin(y*(y + 1.1) + (sin(y)))')
        ('', ('sin(', ('y*(y + 1.1) + (', ('sin(', 'y'), '))')), ')')
        """
    first_par = re.search('(\\w+\\()', exprstr)
    if first_par is None:
        return exprstr
    else:
        start = first_par.start()
        end = first_par.end()
        head = exprstr[:start]
        func = exprstr[start:end]
        tail = exprstr[end:]
        count = 0
        for i, c in enumerate(tail):
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
            if count == -1:
                break
        func_tail = self.str2tree(tail[:i])
        tail = self.str2tree(tail[i:])
        return (head, (func, func_tail), tail)