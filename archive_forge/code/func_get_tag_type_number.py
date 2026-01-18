import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def get_tag_type_number(dtype):
    for number, numpy_dtype in METADATA_DATATYPE.dtypes.items():
        if dtype == numpy_dtype:
            return number
    else:
        return None