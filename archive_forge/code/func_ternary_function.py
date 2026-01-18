import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def ternary_function(ctx, m, x, c):
    mx = pc.call_function('multiply', [m, x], memory_pool=ctx.memory_pool)
    return pc.call_function('add', [mx, c], memory_pool=ctx.memory_pool)