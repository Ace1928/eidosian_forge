from sympy.core import Basic, Integer
import operator
@property
def is_limit_ordinal(self):
    try:
        return not self.trailing_term.exp == ord0
    except ValueError:
        return False