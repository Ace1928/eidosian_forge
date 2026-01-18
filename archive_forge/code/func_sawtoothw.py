from .functions import defun_wrapped
@defun_wrapped
def sawtoothw(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period
    return A * ctx.frac(t / P)