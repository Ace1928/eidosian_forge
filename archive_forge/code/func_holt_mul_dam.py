import numpy as np
def holt_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Multiplicative and Multiplicative Damped
    Minimization Function
    (M,) & (Md,)
    """
    _, beta, phi, alphac, betac, y_alpha = holt_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] + alphac * (lvl[i - 1] * b[i - 1] ** phi)
        b[i] = beta * (lvl[i] / lvl[i - 1]) + betac * b[i - 1] ** phi
    return hw_args.y - lvl * b ** phi