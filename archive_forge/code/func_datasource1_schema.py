import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def datasource1_schema():
    return pa.schema([('', pa.int32()), ('', pa.int32())])