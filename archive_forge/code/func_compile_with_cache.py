import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
def compile_with_cache(*args, **kwargs):
    warnings.warn('cupy.cuda.compile_with_cache has been deprecated in CuPy v10, and will be removed in the future. Use cupy.RawModule or cupy.RawKernel instead.', UserWarning)
    return _compile_module_with_cache(*args, **kwargs)