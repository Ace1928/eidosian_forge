from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_gamma(z, prec, type=0):
    (a1, a2), (b1, b2) = z
    if b1 == b2 == fzero and (type != 3 or mpf_gt(a1, fzero)):
        return (mpi_gamma(z, prec, type), mpi_zero)
    wp = prec + 20
    if type != 3:
        amag = a2[2] + a2[3]
        bmag = b2[2] + b2[3]
        if a2 != fzero:
            mag = max(amag, bmag)
        else:
            mag = bmag
        an = abs(to_int(a2))
        bn = abs(to_int(b2))
        absn = max(an, bn)
        gamma_size = max(0, absn * mag)
        wp += bitcount(gamma_size)
    if type == 1:
        a1, a2 = mpi_add((a1, a2), mpi_one, wp)
        z = ((a1, a2), (b1, b2))
        type = 0
    if mpf_lt(a1, gamma_min_b):
        if mpi_overlap((b1, b2), (gamma_mono_imag_a, gamma_mono_imag_b)):
            znew = (mpi_add((a1, a2), mpi_one, wp), (b1, b2))
            if type == 0:
                return mpci_div(mpci_gamma(znew, prec + 2, 0), z, prec)
            if type == 2:
                return mpci_mul(mpci_gamma(znew, prec + 2, 2), z, prec)
            if type == 3:
                return mpci_sub(mpci_gamma(znew, prec + 2, 3), mpci_log(z, prec + 2), prec)
    if mpf_ge(b1, fzero):
        minre = mpc_loggamma((a1, b2), wp, round_floor)
        maxre = mpc_loggamma((a2, b1), wp, round_ceiling)
        minim = mpc_loggamma((a1, b1), wp, round_floor)
        maxim = mpc_loggamma((a2, b2), wp, round_ceiling)
    elif mpf_le(b2, fzero):
        minre = mpc_loggamma((a1, b1), wp, round_floor)
        maxre = mpc_loggamma((a2, b2), wp, round_ceiling)
        minim = mpc_loggamma((a2, b1), wp, round_floor)
        maxim = mpc_loggamma((a1, b2), wp, round_ceiling)
    else:
        maxre = mpc_loggamma((a2, fzero), wp, round_ceiling)
        if mpf_gt(mpf_neg(b1), b2):
            minre = mpc_loggamma((a1, b1), wp, round_ceiling)
        else:
            minre = mpc_loggamma((a1, b2), wp, round_ceiling)
        minim = mpc_loggamma((a2, b1), wp, round_floor)
        maxim = mpc_loggamma((a2, b2), wp, round_floor)
    w = ((minre[0], maxre[0]), (minim[1], maxim[1]))
    if type == 3:
        return (mpi_pos(w[0], prec), mpi_pos(w[1], prec))
    if type == 2:
        w = mpci_neg(w)
    return mpci_exp(w, prec)