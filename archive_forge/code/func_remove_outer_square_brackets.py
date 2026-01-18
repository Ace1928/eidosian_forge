import re
from . import utilities
def remove_outer_square_brackets(text):
    """
    Checks that test is of the form "[...]" and returns result between
    brackets.

    >>> remove_outer_square_brackets("[a*b*c]")
    'a*b*c'
    >>> remove_outer_square_brackets("[[a,b]")
    '[a,b'
    """
    if text[0] != '[' or text[-1] != ']':
        raise ValueError('Error while parsing: outer square brackets missing')
    return text[1:-1]