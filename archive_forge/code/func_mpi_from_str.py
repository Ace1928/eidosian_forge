from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_from_str(s, prec):
    """
    Parse an interval number given as a string.

    Allowed forms are

    "-1.23e-27"
        Any single decimal floating-point literal.
    "a +- b"  or  "a (b)"
        a is the midpoint of the interval and b is the half-width
    "a +- b%"  or  "a (b%)"
        a is the midpoint of the interval and the half-width
        is b percent of a (`a 	imes b / 100`).
    "[a, b]"
        The interval indicated directly.
    "x[y,z]e"
        x are shared digits, y and z are unequal digits, e is the exponent.

    """
    e = ValueError("Improperly formed interval number '%s'" % s)
    s = s.replace(' ', '')
    wp = prec + 20
    if '+-' in s:
        x, y = s.split('+-')
        return mpi_from_str_a_b(x, y, False, prec)
    elif '(' in s:
        if s[0] == '(' or ')' not in s:
            raise e
        s = s.replace(')', '')
        percent = False
        if '%' in s:
            if s[-1] != '%':
                raise e
            percent = True
            s = s.replace('%', '')
        x, y = s.split('(')
        return mpi_from_str_a_b(x, y, percent, prec)
    elif ',' in s:
        if '[' not in s or ']' not in s:
            raise e
        if s[0] == '[':
            s = s.replace('[', '')
            s = s.replace(']', '')
            a, b = s.split(',')
            a = from_str(a, prec, round_floor)
            b = from_str(b, prec, round_ceiling)
            return (a, b)
        else:
            x, y = s.split('[')
            y, z = y.split(',')
            if 'e' in s:
                z, e = z.split(']')
            else:
                z, e = (z.rstrip(']'), '')
            a = from_str(x + y + e, prec, round_floor)
            b = from_str(x + z + e, prec, round_ceiling)
            return (a, b)
    else:
        a = from_str(s, prec, round_floor)
        b = from_str(s, prec, round_ceiling)
        return (a, b)