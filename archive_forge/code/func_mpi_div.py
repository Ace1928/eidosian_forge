from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_div(s, t, prec):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    if sas == sbs == 0:
        if tas < 0 and tbs > 0 or (tas == 0 or tbs == 0):
            return (fninf, finf)
        return (fzero, fzero)
    if tas < 0 and tbs > 0:
        return (fninf, finf)
    if tas < 0:
        return mpi_div(mpi_neg(s), mpi_neg(t), prec)
    if tas == 0:
        if sas < 0 and sbs > 0:
            return (fninf, finf)
        if tas == tbs:
            return (fninf, finf)
        if sas >= 0:
            a = mpf_div(sa, tb, prec, round_floor)
            b = finf
        if sbs <= 0:
            a = fninf
            b = mpf_div(sb, tb, prec, round_ceiling)
    elif sas >= 0:
        a = mpf_div(sa, tb, prec, round_floor)
        b = mpf_div(sb, ta, prec, round_ceiling)
        if a == fnan:
            a = fzero
        if b == fnan:
            b = finf
    elif sbs <= 0:
        a = mpf_div(sa, ta, prec, round_floor)
        b = mpf_div(sb, tb, prec, round_ceiling)
        if a == fnan:
            a = fninf
        if b == fnan:
            b = fzero
    else:
        a = mpf_div(sa, ta, prec, round_floor)
        b = mpf_div(sb, ta, prec, round_ceiling)
        if a == fnan:
            a = fninf
        if b == fnan:
            b = finf
    return (a, b)