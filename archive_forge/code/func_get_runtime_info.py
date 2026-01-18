import inspect
import io
import os
import platform
import warnings
import numpy
import cupy
import cupy_backends
def get_runtime_info(*, full=True):
    return _RuntimeInfo(full=full)