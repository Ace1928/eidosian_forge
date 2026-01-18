import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def sum_agg_func_fixture():
    """
    Register a unary aggregate function (mean)
    """

    def func(ctx, x, *args):
        return pa.scalar(np.nansum(x))
    func_name = 'sum_udf'
    func_doc = empty_udf_doc
    pc.register_aggregate_function(func, func_name, func_doc, {'x': pa.float64()}, pa.float64())
    return (func, func_name)