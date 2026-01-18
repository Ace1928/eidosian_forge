import contextlib
import functools
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import setuptools
@functools.lru_cache()
def cuda_include_dir():
    base_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    cuda_path = os.path.join(base_dir, 'third_party', 'cuda')
    return os.path.join(cuda_path, 'include')