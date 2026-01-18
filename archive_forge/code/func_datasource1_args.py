import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def datasource1_args(func, func_name):
    func_doc = {'summary': f'{func_name} UDT', 'description': 'test {func_name} UDT'}
    in_types = {}
    out_type = pa.struct([('', pa.int32()), ('', pa.int32())])
    return (func, func_name, func_doc, in_types, out_type)