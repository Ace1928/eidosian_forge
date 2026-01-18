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
def test_generic_ext_type_equality():
    period_type = PeriodType('D')
    assert period_type.extension_name == 'test.period'
    period_type2 = PeriodType('D')
    period_type3 = PeriodType('H')
    assert period_type == period_type2
    assert not period_type == period_type3