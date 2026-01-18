from .functions import defun, defun_wrapped
def comp_fp_tolerance(ctx, n):
    wpz = wpzeros(n * ctx.log(n))
    if n < 15 * 10 ** 8:
        fp_tolerance = 0.0005
    elif n <= 10 ** 14:
        fp_tolerance = 0.1
    else:
        fp_tolerance = 100
    return (wpz, fp_tolerance)