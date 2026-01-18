import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def localeconv():
    """ localeconv() -> dict.
            Returns numeric and monetary locale-specific parameters.
        """
    return {'grouping': [127], 'currency_symbol': '', 'n_sign_posn': 127, 'p_cs_precedes': 127, 'n_cs_precedes': 127, 'mon_grouping': [], 'n_sep_by_space': 127, 'decimal_point': '.', 'negative_sign': '', 'positive_sign': '', 'p_sep_by_space': 127, 'int_curr_symbol': '', 'p_sign_posn': 127, 'thousands_sep': '', 'mon_thousands_sep': '', 'frac_digits': 127, 'mon_decimal_point': '', 'int_frac_digits': 127}