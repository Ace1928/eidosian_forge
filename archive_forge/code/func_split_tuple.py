import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if len(expr) > 2 and expr[0] in TUPLE_CHARS and (expr[-1] in TUPLE_CHARS) and all([ch not in expr[1:-1] for ch in TUPLE_CHARS]):
        elems = [elem.strip() for elem in expr[1:-1].split(',')]
    else:
        elems = [expr]
    return elems