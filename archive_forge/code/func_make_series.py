import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
def make_series(context, builder, typ, **kwargs):
    return cgutils.create_struct_proxy(typ)(context, builder, **kwargs)