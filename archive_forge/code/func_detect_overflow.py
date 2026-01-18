import collections
from .utils import ExplicitEnum, is_torch_available, logging
def detect_overflow(var, ctx):
    """
    Report whether the tensor contains any `nan` or `inf` entries.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the tensor in question.

    This function contains a few other helper features that you can enable and tweak directly if you want to track
    various other things.

    Args:
        var: the tensor variable to check
        ctx: the message to print as a context

    Return:
        `True` if `inf` or `nan` was detected, `False` otherwise
    """
    detected = False
    if torch.isnan(var).any().item():
        detected = True
        print(f'{ctx} has nans')
    if torch.isinf(var).any().item():
        detected = True
        print(f'{ctx} has infs')
    if 0:
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            print(f'{ctx}:  n100={n100.numel()}')
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            print(f'{ctx}: n1000={n1000.numel()}')
        n10000 = var[torch.ge(var.abs(), 10000)]
        if n10000.numel() > 0:
            print(f'{ctx}: n10000={n10000.numel()}')
    if 0:
        print(f'min={var.min():9.2e} max={var.max():9.2e}')
    if 0:
        print(f'min={var.min():9.2e} max={var.max():9.2e} var={var.var():9.2e} mean={var.mean():9.2e} ({ctx})')
    return detected