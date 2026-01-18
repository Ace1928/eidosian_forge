import numpy as np
def holt_win_mul_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped with Multiplicative Seasonal
    Minimization Function
    (M,M) & (Md,M)
    """
    _, beta, _, phi, alphac, betac, gammac, y_alpha, y_gamma = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    b = hw_args.b
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] / s[i - 1] + alphac * (lvl[i - 1] * b[i - 1] ** phi)
        b[i] = beta * (lvl[i] / lvl[i - 1]) + betac * b[i - 1] ** phi
        s[i + m - 1] = y_gamma[i - 1] / (lvl[i - 1] * b[i - 1] ** phi) + gammac * s[i - 1]
    return hw_args.y - lvl * b ** phi * s[:-(m - 1)]