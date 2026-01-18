import collections
from .utils import ExplicitEnum, is_torch_available, logging
def get_abs_min_max(var, ctx):
    abs_var = var.abs()
    return f'{abs_var.min():8.2e} {abs_var.max():8.2e} {ctx}'