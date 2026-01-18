from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def plot_with_error(y, error, x=None, axes=None, value_fmt='k', error_fmt='k--', alpha=0.05, stderr_type='asym'):
    """
    Make plot with optional error bars

    Parameters
    ----------
    y :
    error : array or None
    """
    import matplotlib.pyplot as plt
    if axes is None:
        axes = plt.gca()
    x = x if x is not None else lrange(len(y))
    plot_action = lambda y, fmt: axes.plot(x, y, fmt)
    plot_action(y, value_fmt)
    if error is not None:
        if stderr_type == 'asym':
            q = util.norm_signif_level(alpha)
            plot_action(y - q * error, error_fmt)
            plot_action(y + q * error, error_fmt)
        if stderr_type in ('mc', 'sz1', 'sz2', 'sz3'):
            plot_action(error[0], error_fmt)
            plot_action(error[1], error_fmt)