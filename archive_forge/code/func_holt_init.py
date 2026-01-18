import numpy as np
def holt_init(x, hw_args: HoltWintersArgs):
    """
    Initialization for the Holt Models
    """
    hw_args.p[hw_args.xi.astype(bool)] = x
    if hw_args.transform:
        alpha, beta, _ = to_restricted(hw_args.p, hw_args.xi, hw_args.bounds)
    else:
        alpha, beta = hw_args.p[:2]
    l0, b0, phi = hw_args.p[3:6]
    alphac = 1 - alpha
    betac = 1 - beta
    y_alpha = alpha * hw_args.y
    hw_args.lvl[0] = l0
    hw_args.b[0] = b0
    return (alpha, beta, phi, alphac, betac, y_alpha)