from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def formula_to_composition(formula, prefixes=None, suffixes=('(s)', '(l)', '(g)', '(aq)')):
    """Parse composition of formula representing a chemical formula

    Composition is represented as a dict mapping int -> int (atomic
    number -> multiplicity). "Atomic number" 0 represents net charge.

    Parameters
    ----------
    formula: str
        Chemical formula, e.g. 'H2O', 'Fe+3', 'Cl-'
    prefixes: iterable strings
        Prefixes to ignore, e.g. ('.', 'alpha-')
    suffixes: tuple of strings
        Suffixes to ignore, e.g. ('(g)', '(s)')

    Examples
    --------
    >>> formula_to_composition('NH4+') == {0: 1, 1: 4, 7: 1}
    True
    >>> formula_to_composition('.NHO-(aq)') == {0: -1, 1: 1, 7: 1, 8: 1}
    True
    >>> formula_to_composition('Na2CO3..7H2O') == {11: 2, 6: 1, 8: 10, 1: 14}
    True

    """
    if prefixes is None:
        prefixes = _latex_mapping.keys()
    stoich_tok, chg_tok = _formula_to_parts(formula, prefixes, suffixes)[:2]
    tot_comp = {}
    parts = stoich_tok.split('..')
    for idx, stoich in enumerate(parts):
        if idx == 0:
            m = 1
        else:
            m, stoich = _get_leading_integer(stoich)
        comp = _parse_stoich(stoich)
        for k, v in comp.items():
            if k not in tot_comp:
                tot_comp[k] = m * v
            else:
                tot_comp[k] += m * v
    if chg_tok is not None:
        tot_comp[0] = _get_charge(chg_tok)
    return tot_comp