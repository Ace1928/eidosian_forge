from sympy.core import Basic, Integer
import operator
@property
def is_successor_ordinal(self):
    try:
        return self.trailing_term.exp == ord0
    except ValueError:
        return False