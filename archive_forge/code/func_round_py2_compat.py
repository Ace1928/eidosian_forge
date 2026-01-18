import sys
import decimal
from decimal import Context
def round_py2_compat(what):
    """
    Python 2 and Python 3 use different rounding strategies in round(). This
    function ensures that results are python2/3 compatible and backward
    compatible with previous py2 releases
    :param what: float
    :return: rounded long
    """
    d = Context(prec=len(str(long(what))), rounding=decimal.ROUND_HALF_UP).create_decimal(str(what))
    return long(d)