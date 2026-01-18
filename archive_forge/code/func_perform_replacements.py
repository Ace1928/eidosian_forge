from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
def perform_replacements(fstr, values):
    """Replace placeholders in an fstring with subexpressions."""
    for i, value in enumerate(values):
        fstr = fstr.replace(_wrap(placeholder(i)), _wrap(value))
    return fstr