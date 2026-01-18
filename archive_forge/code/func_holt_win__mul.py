import numpy as np
def holt_win__mul(x, hw_args: HoltWintersArgs):
    """
    Multiplicative Seasonal
    Minimization Function
    (,M)
    """
    _, _, _, _, alphac, _, gammac, y_alpha, y_gamma = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] / s[i - 1] + alphac * lvl[i - 1]
        s[i + m - 1] = y_gamma[i - 1] / lvl[i - 1] + gammac * s[i - 1]
    return hw_args.y - lvl * s[:-(m - 1)]