from .functions import defun, defun_wrapped
@defun
def airyaizero(ctx, k, derivative=0):
    return _airy_zero(ctx, 0, k, derivative, False)