import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.fixture(params=[PeriodType('D'), PeriodTypeWithClass('D'), PeriodTypeWithToPandasDtype('D')])
def registered_period_type(request):
    period_type = request.param
    period_class = period_type.__arrow_ext_class__()
    pa.register_extension_type(period_type)
    yield (period_type, period_class)
    try:
        pa.unregister_extension_type('test.period')
    except KeyError:
        pass