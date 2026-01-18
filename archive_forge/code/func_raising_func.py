import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def raising_func(ctx):
    raise MyError('error raised by scalar UDF')