from .functions import defun, defun_wrapped
@defun
def airybizero(ctx, k, derivative=0, complex=False):
    return _airy_zero(ctx, 1, k, derivative, complex)