from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_mul(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    if sas == sbs == 0:
        if ta == fninf or tb == finf:
            return (fninf, finf)
        return (fzero, fzero)
    if tas == tbs == 0:
        if sa == fninf or sb == finf:
            return (fninf, finf)
        return (fzero, fzero)
    if sas >= 0:
        if tas >= 0:
            a = mpf_mul(sa, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan:
                a = fzero
            if b == fnan:
                b = finf
        elif tbs <= 0:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sa, tb, prec, round_ceiling)
            if a == fnan:
                a = fninf
            if b == fnan:
                b = fzero
        else:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan:
                a = fninf
            if b == fnan:
                b = finf
    elif sbs <= 0:
        if tas >= 0:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sb, ta, prec, round_ceiling)
            if a == fnan:
                a = fninf
            if b == fnan:
                b = fzero
        elif tbs <= 0:
            a = mpf_mul(sb, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan:
                a = fzero
            if b == fnan:
                b = finf
        else:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan:
                a = fninf
            if b == fnan:
                b = finf
    else:
        cases = [mpf_mul(sa, ta), mpf_mul(sa, tb), mpf_mul(sb, ta), mpf_mul(sb, tb)]
        if fnan in cases:
            a, b = (fninf, finf)
        else:
            a, b = mpf_min_max(cases)
            a = mpf_pos(a, prec, round_floor)
            b = mpf_pos(b, prec, round_ceiling)
    return (a, b)