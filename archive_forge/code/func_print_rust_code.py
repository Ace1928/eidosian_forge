from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def print_rust_code(expr, **settings):
    """Prints Rust representation of the given expression."""
    print(rust_code(expr, **settings))