from .exceptions import ParseException
from .util import col, replaced_by_pep8
def match_only_at_col(n):
    """
    Helper method for defining parse actions that require matching at
    a specific column in the input text.
    """

    def verify_col(strg, locn, toks):
        if col(locn, strg) != n:
            raise ParseException(strg, locn, f'matched token not at column {n}')
    return verify_col