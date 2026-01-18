import doctest
import re
import decimal
def init_precisions(precisions):
    """
    Register flags for given precisions with doctest module.

    Unfortunately, doctest doesn't seem to support a more generic mechanism
    such as "# doctest: +NUMERIC: 6" to specify the precision and we need to
    unroll each precision we want to its own flag.
    """
    global NUMERIC_LIST
    global NUMERIC_DICT
    global ALL_NUMERIC
    for precision in precisions:
        if precision not in NUMERIC_DICT:
            flag = doctest.register_optionflag('NUMERIC%d' % precision)
            NUMERIC_LIST.append((precision, flag))
            NUMERIC_DICT[precision] = flag
            ALL_NUMERIC |= flag