import numpy as np
def holt_win__add(x, hw_args: HoltWintersArgs):
    """
    Additive Seasonal
    Minimization Function
    (,A)
    """
    alpha, _, gamma, _, alphac, _, gammac, y_alpha, y_gamma = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] - alpha * s[i - 1] + alphac * lvl[i - 1]
        s[i + m - 1] = y_gamma[i - 1] - gamma * lvl[i - 1] + gammac * s[i - 1]
    return hw_args.y - lvl - s[:-(m - 1)]