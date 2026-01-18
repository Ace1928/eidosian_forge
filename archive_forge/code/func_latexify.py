from __future__ import annotations
import re
from fractions import Fraction
def latexify(formula: str, bold: bool=False):
    """Generates a LaTeX formatted formula. E.g., Fe2O3 is transformed to
    Fe$_{2}$O$_{3}$.

    Note that Composition now has a to_latex_string() method that may
    be used instead.

    Args:
        formula (str): Input formula.
        bold (bool): Whether to make the subscripts bold. Defaults to False.

    Returns:
        Formula suitable for display as in LaTeX with proper subscripts.
    """
    return re.sub('([A-Za-z\\(\\)])([\\d\\.]+)', '\\1$_{\\\\mathbf{\\2}}$' if bold else '\\1$_{\\2}$', formula)