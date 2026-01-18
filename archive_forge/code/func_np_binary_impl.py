import math
import numpy as np
from functools import lru_cache
from numba.core import typing
from numba.cuda.mathimpl import (get_unary_impl_for_fn_and_ty,
def np_binary_impl(fn, context, builder, sig, args):
    npyfuncs._check_arity_and_homogeneity(sig, args, 2)
    impl = get_binary_impl_for_fn_and_ty(fn, sig.args[0])
    return impl(context, builder, sig, args)